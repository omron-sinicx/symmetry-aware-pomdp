import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu

from escnn import nn as enn

class Actor_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_type,
        algo,
        action_embedding_size,
        observ_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        group_helper=None,
        image_encoder=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.group_helper = group_helper

        if actor_type in ['normal', 'aug-aux', 'aug', 'aux']:
            self.embed_type = 'normal'
        elif actor_type in ['equi']:
            self.embed_type = 'equi'
        else:
            raise ValueError("Embed type not supported.")

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            if self.embed_type == 'normal':
                self.observ_embedder = utl.FeatureExtractor(
                    obs_dim, observ_embedding_size, F.relu
                )
            else:
                num_rotations = self.group_helper.num_rotations
                grp_act = self.group_helper.grp_act
                in_type = self.group_helper.obs_in_type
                out_type = enn.FieldType(grp_act, observ_embedding_size //
                                         num_rotations * self.group_helper.reg_repr)

                self.observ_embedder = utl.EquiFeatureExtractor(in_type, out_type)
        else:  # for pixel observation, use external encoder
            assert observ_embedding_size == 0
            observ_embedding_size = self.image_encoder.embed_size  # reset it

        if self.embed_type == 'normal':
            self.action_embedder = utl.FeatureExtractor(
                action_dim, action_embedding_size, F.relu
            )
        else:
            num_rotations = self.group_helper.num_rotations
            grp_act = self.group_helper.grp_act
            in_type = self.group_helper.act_in_type
            out_type = enn.FieldType(grp_act, action_embedding_size // num_rotations * self.group_helper.reg_repr)

            self.action_embedder = utl.EquiFeatureExtractor(in_type, out_type)

        ## 2. build RNN model
        rnn_input_size = (
            action_embedding_size + observ_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        if actor_type in ['normal', 'aug-aux', 'aug', 'aux']:
            encoder = 'lstm'
        elif actor_type in ['equi']:
            encoder = 'equi-conv-lstm'
        else:
            raise ValueError("actor_type not type not supported.")

        assert encoder in RNNs
        self.encoder = encoder
        self.num_layers = rnn_num_layers

        self.rnn = RNNs[encoder](
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bias=True,
        )

        if actor_type == 'equi' and isinstance(self.rnn, EqConvLSTM):
            self.rnn.init_group(group_helper)

        # never add activation after GRU cell, cuz the last operation of GRU is tanh

        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.rnn.named_parameters():
            if 'equi' not in self.encoder:
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

        ## 3. build another obs branch
        if self.image_encoder is None:
            if self.embed_type == 'normal':
                input_dim = obs_dim
                self.current_observ_embedder = utl.FeatureExtractor(
                    input_dim, observ_embedding_size, F.relu
                )
            else:
                num_rotations = self.group_helper.num_rotations
                grp_act = self.group_helper.grp_act
                in_type = self.group_helper.obs_in_type
                out_type = enn.FieldType(grp_act, observ_embedding_size //
                                         num_rotations * self.group_helper.reg_repr)

                self.current_observ_embedder = utl.EquiFeatureExtractor(in_type, out_type)

        ## 4. build policy
        if actor_type in ['normal', 'aug-aux', 'aug', 'aux']:
            policy_build_function = self.algo.build_actor
        elif actor_type in ['equi']:
            policy_build_function = self.algo.build_equi_actor
        else:
            raise ValueError("actor_type not supported.")
        self.policy = policy_build_function(
            input_size=self.rnn_hidden_size + observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
            group_helper=self.group_helper,
        )

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return self.observ_embedder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def _get_shortcut_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return self.current_observ_embedder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def get_hidden_states(
        self, prev_actions, observs, initial_internal_state=None
    ):
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            output, _ = self.rnn(inputs)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs,
                                                      initial_internal_state)
            return output, current_internal_state

    def forward(self, prev_actions, observs):
        """
        For prev_actions a, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == observs.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with states
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, observs=observs
        )

        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(observs)  # (T+1, B, dim)

        # 3. joint embed
        seq_len = prev_actions.shape[0]
        bs = prev_actions.shape[1]
        hidden_states = utl.process_hidden(hidden_states, seq_len, bs,
                                           self.encoder)

        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)  # (T+1, B, dim)

        # 4. Actor
        return self.algo.forward_actor(actor=self.policy, observ=joint_embeds)

    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()

        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        elif self.encoder in [ConvLSTM_name, EqConvLSTM_name]:
            hidden_state = ptu.zeros((self.num_layers, self.rnn_hidden_size, 1, 1)).float()
            cell_state = ptu.zeros((self.num_layers, self.rnn_hidden_size, 1, 1)).float()
            internal_state = (hidden_state, cell_state)
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            internal_state = (hidden_state, cell_state)

        return prev_action, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module

        # 1. get hidden state and current internal state
        ## NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (layers, B, dim) or ((layers, B, dim), (layers, B, dim))
        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            observs=obs,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(obs)  # (1, B, dim)

        # 3. joint embed
        hidden_state = utl.process_hidden(hidden_state, 1, 1, self.encoder)
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        action_tuple = self.algo.select_action(
            actor=self.policy,
            observ=joint_embeds,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return action_tuple, current_internal_state

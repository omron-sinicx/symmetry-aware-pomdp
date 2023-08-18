import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
from argparse import ArgumentParser

from policies.models.recurrent_critic import Critic_RNN
from policies.rl.sac import SAC
from utils import helpers as utl
from utils.group import GroupHelper

from escnn import nn as enn

import envs.pomdp

cuda_id = 0  # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

action_embedding_size = 20
rnn_hidden_size = 125
observ_embedding_size = 30
policy_layers = [128, 128]
rnn_num_layers = 1

seq_len = 51
batch_size = 32

def test_critics(domain, group):

    group_helper = GroupHelper(group, domain, "equi", "equi")

    env = gym.make(domain)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    critic = Critic_RNN(
        obs_dim,
        act_dim,
        0,
        "equi",
        SAC,
        action_embedding_size,
        observ_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        group_helper=group_helper,
    ).to(ptu.device)

    utl.count_parameters(critic)

    seq_len = 1
    batch_size = 1

    acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
    curr_acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
    obses = torch.randn(seq_len, batch_size, obs_dim).to(ptu.device)

    q1, q2 = critic(acts, obses, curr_acts)
    q1_or = q1
    q2_or = q2

    grp_act = group_helper.grp_act

    acts = acts.reshape((seq_len*batch_size, act_dim, 1, 1))
    curr_acts = curr_acts.reshape((seq_len*batch_size, act_dim, 1, 1))
    obses = obses.reshape((seq_len*batch_size, obs_dim, 1, 1))

    acts = group_helper.act_in_type(acts)
    curr_acts = group_helper.act_in_type(curr_acts)
    obses = group_helper.obs_in_type(obses)

    # for each group element
    with torch.no_grad():
        for g in grp_act.testing_elements:
            acts_transformed = acts.transform(g)
            curr_acts_transformed = curr_acts.transform(g)
            obses_transformed = obses.transform(g)

            acts_transformed = acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
            curr_acts_transformed = curr_acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
            obses_transformed = obses_transformed.tensor.reshape((seq_len, batch_size, obs_dim))

            q1_new, q2_new = critic(acts_transformed, obses_transformed, curr_acts_transformed)

            assert torch.allclose(q1_or, q1_new, atol=1e-5), g
            assert torch.allclose(q2_or, q2_new, atol=1e-5), g


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="PegInsertion-Square-XYZ-v0")
    parser.add_argument('--group', type=str, default="FlipXY")
    args = parser.parse_args()

    test_critics(args.domain, args.group)
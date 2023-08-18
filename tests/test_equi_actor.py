import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
from argparse import ArgumentParser

from policies.models.recurrent_actor import Actor_RNN
from policies.rl.sac import SAC
from utils import helpers as utl
from utils.group import GroupHelper

from escnn import gspaces
from escnn import nn as enn

import envs.pomdp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cuda_id = 0  # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

action_embedding_size = 20
rnn_hidden_size = 125
observ_embedding_size = 30
policy_layers = [128, 128]
rnn_num_layers = 1

seq_len = 51
batch_size = 32


def test_actor(domain, group):

    group_helper = GroupHelper(group, domain, "equi", "equi")

    env = gym.make(domain)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    equi_actor = Actor_RNN(
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

    utl.count_parameters(equi_actor)

    acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
    obses = torch.randn(seq_len, batch_size, obs_dim).to(ptu.device)

    _, _, mean, log_stds = equi_actor(acts, obses)  # 51, 32, 4

    mean = mean.reshape([seq_len*batch_size, act_dim, 1, 1])
    log_stds = log_stds.reshape([seq_len*batch_size, act_dim, 1, 1])

    mean_or = group_helper.mean_out_type(mean)
    log_stds_or = group_helper.logstds_out_type(log_stds)

    acts = acts.reshape((seq_len*batch_size, act_dim, 1, 1)).to(ptu.device)
    obses = obses.reshape((seq_len*batch_size, obs_dim, 1, 1)).to(ptu.device)

    acts = group_helper.act_in_type(acts)

    obses = group_helper.obs_in_type(obses)

    # for each group element
    with torch.no_grad():
        for g in group_helper.grp_act.testing_elements:
            acts_transformed = acts.transform(g)
            obses_transformed = obses.transform(g)

            acts_transformed = acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
            obses_transformed = obses_transformed.tensor.reshape((seq_len, batch_size, obs_dim))

            _, _, mean_new, log_stds_new = equi_actor(acts_transformed, obses_transformed)

            mean = mean_or.transform(g).tensor.reshape_as(mean_new)
            log_stds = log_stds_or.transform(g).tensor.reshape_as(log_stds_new)

            assert torch.allclose(mean_new, mean, atol=1e-5), g
            assert torch.allclose(log_stds_new, log_stds, atol=1e-5), g


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="PegInsertion-Square-Old-XYZ-v0")
    parser.add_argument('--group', type=str, default="FlipXY")
    args = parser.parse_args()

    test_actor(args.domain, args.group)


import numpy as np
import robosuite as suite

import argparse

parser = argparse.ArgumentParser(description='Robosuite test')
parser.add_argument('--robot', type=str, default='SoftUR5eRound')
parser.add_argument('--zero-action', action='store_true')
parser.add_argument('--task', type=str, default='SoftPegInHole')

args = parser.parse_args()

# create environment instance
env = suite.make(
    env_name=args.task, # try with other tasks like "Stack" and "Door"
    robots=args.robot,  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    if args.zero_action:
        action = np.zeros(env.robots[0].dof)
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
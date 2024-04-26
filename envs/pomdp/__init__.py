from gym.envs.registration import register
import pdomains


# POMDPs
register(
    "PegInsertion-Square-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square"}
)

register(
    "PegInsertion-Triangle-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "triangle"}
)

register(
    "PegInsertion-Pentagon-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "pentagon"}
)

register(
    "PegInsertion-Hexagon-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "hexagon"}
)

register(
    "PegInsertion-Round-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "round"}
)


# MDPs
register(
    "PegInsertion-Triangle-State-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "triangle", "return_state": True}
)

register(
    "PegInsertion-Square-State-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "square", "return_state": True}
)

register(
    "PegInsertion-Pentagon-State-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "pentagon", "return_state": True}
)

register(
    "PegInsertion-Hexagon-State-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "hexagon", "return_state": True}
)

register(
    "PegInsertion-Round-State-XYZ-v0",
    entry_point='pdomains.peg_insertion_xyz:PegInsertionEnv',
    max_episode_steps=200,
    kwargs={"peg_type": "round", "return_state": True}
)

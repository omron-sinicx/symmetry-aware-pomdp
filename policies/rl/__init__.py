from .sac import SAC
from .td3 import TD3

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
}

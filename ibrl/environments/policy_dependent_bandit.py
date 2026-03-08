import numpy as np
from numpy.typing import NDArray

from . import BaseNewcombLikeEnvironment


class PolicyDependentBanditEnvironment(BaseNewcombLikeEnvironment):
    """
    Like a multi-armed bandit, but the reward depends on the (prediction,action) pair, rather than just the action.
    This is a generalisation of both the multi-armed bandit and Newcomb-like problems
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reward_table=[[0,0],[0,0]], **kwargs)

    def reset(self):
        super().reset()
        self.reward_table = self.random.standard_normal((self.num_actions,self.num_actions))  # 2D array

import numpy as np
from numpy.typing import NDArray

from . import BaseNewcombLikeEnvironment


class CoordinationGameEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *args,
            rewardA : float = 2,  # reward in first equilibrium
            rewardB : float = 1,  # reward in second equilibrium
            **kwargs):
        super().__init__(*args, reward_table=[
            [rewardA, 0      ],
            [0,       rewardB],
        ], **kwargs)

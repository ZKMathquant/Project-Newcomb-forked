import numpy as np
from numpy.typing import NDArray

from . import BaseNewcombLikeEnvironment


class DeathInDamascusEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *args,
        death : float =  0,  # reward upon death
        life  : float = 10,  # reward upon survival
        **kwargs):
        super().__init__(*args, reward_table=[
            [death, life ],
            [life,  death],
        ], **kwargs)

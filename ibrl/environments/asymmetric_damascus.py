import numpy as np
from numpy.typing import NDArray

from . import BaseNewcombLikeEnvironment


class AsymmetricDeathInDamascusEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *args,
            death_in_damascus : float =  0,  # reward upon death in Damascus
            death_in_aleppo   : float =  5,  # reward upon death in Aleppo
            life              : float = 10,  # reward upon survival
            **kwargs):
        super().__init__(*args, reward_table=[
            [death_in_damascus, life           ],
            [life,              death_in_aleppo],
        ], **kwargs)

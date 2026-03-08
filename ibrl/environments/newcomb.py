import numpy as np
from numpy.typing import NDArray

from . import BaseNewcombLikeEnvironment


class NewcombEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *args,
        boxA : float =  5,  # guaranteed content of first box
        boxB : float = 10,  # conditional content of second box
        **kwargs):
        super().__init__(*args, reward_table=[
            [boxB, boxB+boxA],
            [0,    boxA     ]
        ], **kwargs)

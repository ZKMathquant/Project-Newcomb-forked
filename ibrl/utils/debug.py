import numpy as np
from numpy.typing import NDArray


def dump_array(array : NDArray[np.float64], format="%.2f") -> str:
    """
    Short string representation of 1D array for debugging
    """
    return "["+",".join(format%x for x in array)+"]"

"""A-measure class — direct port from coin-learning.ipynb."""
import numpy as np
from numpy.typing import NDArray


class AMeasure:
    """An a-measure, characterised by a scale factor lambda > 0,
    a probability measure mu over the history space, and an offset b >= 0.
    """
    def __init__(self, measure: NDArray[np.float64], *, history_space: list,
                 scale: float = 1.0, offset: float = 0.0):
        assert len(measure) == len(history_space)
        assert measure.min() >= 0
        self.measure = measure.copy().astype(np.float64)
        self.history_space = history_space
        self.scale = scale
        self.offset = offset
        self._normalise_measure()

    def expected_value(self, reward_function: callable) -> float:
        """Compute alpha(f) = lambda * mu(f) + b."""
        value = 0.0
        for i, x in enumerate(self.history_space):
            value += self.measure[i] * reward_function(x)
        return self.scale * value + self.offset

    def chop(self, pattern) -> None:
        """Set measure to 0 for histories inconsistent with observation."""
        from .helpers import match
        for i, x in enumerate(self.history_space):
            if not match(pattern, x):
                self.measure[i] = 0
        self._normalise_measure()

    def _normalise_measure(self) -> None:
        """Ensure mu(1) == 1, absorb normalisation into scale."""
        total = self.measure.sum()
        if total > 0:
            self.measure /= total
            self.scale *= total

    def __add__(self, other):
        assert isinstance(other, AMeasure)
        return AMeasure(
            self.scale * self.measure + other.scale * other.measure,
            history_space=self.history_space,
            scale=1.0,
            offset=self.offset + other.offset,
        )

    def __itruediv__(self, other: float):
        self.scale /= other
        self.offset /= other
        return self

    def __truediv__(self, other: float):
        return AMeasure(
            self.measure.copy(),
            history_space=self.history_space,
            scale=self.scale / other,
            offset=self.offset / other,
        )

    def is_valid(self, epsilon=1e-6):
        """Check if this is a valid a-measure: alpha(1) <= 1."""
        alpha_one = self.scale + self.offset  # mu(1)=1 by normalisation, so alpha(1) = scale + offset
        return alpha_one <= 1.0 + epsilon

    def __repr__(self) -> str:
        return f"({self.scale:.3f}[{','.join('%.3f' % x for x in self.measure)}],{self.offset:.3f})"

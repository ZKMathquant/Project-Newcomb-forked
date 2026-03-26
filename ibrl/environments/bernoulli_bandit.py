import numpy as np

from . import BaseEnvironment


class BernoulliBanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit with Bernoulli (coin-flip) rewards.

    Each arm has a fixed probability p_i of paying reward 1; otherwise reward is 0.
    Probabilities are sampled uniformly from [0, 1] on reset.
    """
    def _resolve(self, env_action: int | None, action: int) -> float:
        return float(self.random.random() < self.probs[action])

    def get_optimal_reward(self) -> float:
        return self.probs.max()

    def reset(self):
        super().reset()
        self.probs = self.random.random((self.num_actions,))

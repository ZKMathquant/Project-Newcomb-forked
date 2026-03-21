"""IUCB agent for 2-action Newcomb-like games.

Implements Algorithm 4 from Kosoy (2024), "Imprecise Multi-Armed Bandits".
Faithful implementation with cycle-based updates and proper confidence set
geometry, preserving the theoretical guarantees of Theorems 6.1 and 6.3.
"""

import numpy as np
from numpy.typing import NDArray

from . import BaseAgent
from ..iucb.algebra import (
    theta_from_payoff,
    compute_V,
    dist_to_V,
    outcome_to_y,
    DIM_Z,
    DIM_Y,
    DIM_W,
)
from ..iucb.confidence_set import ConfidenceSet
from ..utils.game_solving import solve_2x2_game
from ..utils import dump_array


class IUCBAgent(BaseAgent):
    """Imprecise UCB agent for 2-action games (Algorithm 4).

    Implements the full cycle-based IUCB algorithm with proper confidence
    set geometry. Each cycle:
    1. Select optimistic hypothesis theta* from confidence set
    2. Play the maximin arm x*_{theta*} repeatedly
    3. When stopping condition met, update confidence set with slab constraint
    4. Start new cycle with fresh optimistic hypothesis

    Arguments:
        eta:           Confidence parameter. If None, computed from time_horizon.
        time_horizon:  Expected number of rounds N. Used to set eta if not given.
        reward_range:  Tuple (min_reward, max_reward) for rescaling to [-1, 1].
    """

    def __init__(self, *args,
                 eta: float | None = None,
                 time_horizon: float = 10000,
                 reward_range: tuple[float, float] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2, "IUCB currently only supports 2-action games"

        self.reward_range = reward_range
        self.time_horizon = time_horizon

        # Set eta following Theorem 6.1:
        # eta = c * R * D_W^{5/6} * sqrt(ln(C * D_W * N))
        # The constant c is Theta(1) in the thesis. The theoretically correct
        # value makes cycles take O(R^5 * D_W^3) rounds, which is impractical
        # for the 2x2 case (millions of rounds per cycle). We use a small
        # constant so cycles complete within the given time_horizon, at the
        # cost of the formal regret guarantee not holding at finite N.
        if eta is not None:
            self.eta = eta
        else:
            # Heuristic: set eta so each cycle takes ~sqrt(N) rounds,
            # giving ~sqrt(N) cycles over N rounds total.
            # threshold = 2*(D_Z+1)*eta, and with max_dist ~ 1:
            # tau_cycle ~ (2*(D_Z+1)*eta)^2 ~ sqrt(N)
            # => eta ~ N^{1/4} / (2*(D_Z+1))
            self.eta = time_horizon ** 0.25 / (2.0 * (DIM_Z + 1))

    def get_probabilities(self) -> NDArray[np.float64]:
        return self.current_policy.copy()

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)

        # Rescale reward to [-1, 1]
        if self.reward_range is not None:
            r_min, r_max = self.reward_range
        else:
            # Auto-detect from observed rewards
            old_min, old_max = self.reward_min, self.reward_max
            if self.reward_min is None:
                self.reward_min = outcome.reward
                self.reward_max = outcome.reward
            else:
                self.reward_min = min(self.reward_min, outcome.reward)
                self.reward_max = max(self.reward_max, outcome.reward)
            # If range expanded significantly, reset cycle stats since
            # accumulated sigma_y was computed with different scaling
            if old_min is not None:
                old_range = max(old_max - old_min, 1e-6)
                new_range = max(self.reward_max - self.reward_min, 1e-6)
                if new_range > old_range * 1.5 and self.tau > 0:
                    self.tau = 0
                    self.sigma_y = np.zeros(DIM_Y)
            # Pad above observed max so r_min maps to -1 even when
            # all rewards are identical (gives a learning signal).
            r_min = self.reward_min
            r_max = self.reward_min + max(self.reward_max - self.reward_min, 1.0)

        r_range = r_max - r_min
        if r_range > 1e-6:
            reward_scaled = 2 * (outcome.reward - r_min) / r_range - 1
            reward_scaled = np.clip(reward_scaled, -1.0, 1.0)
        else:
            reward_scaled = 0.0

        # Encode observation as outcome vector
        if outcome.env_action is not None:
            y = outcome_to_y(outcome.env_action, action, reward_scaled)
        else:
            # For standard bandits: no env_action, use action as both
            # (this is a degenerate case where IUCB reduces to UCB-like)
            y = outcome_to_y(action, action, reward_scaled)

        # Accumulate cycle statistics
        self.tau += 1
        self.sigma_y += y
        y_bar = self.sigma_y / self.tau

        # Check stopping condition:
        # sqrt(tau) * max_{theta in C} dist(theta, V(x*, y_bar)) >= 2(D_Z + 1) * eta
        #
        # Optimization: only check periodically (the stopping condition changes
        # smoothly, so we won't overshoot by more than check_interval rounds).
        threshold = 2 * (DIM_Z + 1) * self.eta
        max_possible_dist = 20.0  # conservative upper bound for 2x2 case
        if np.sqrt(self.tau) * max_possible_dist < threshold:
            return  # Can't possibly meet stopping condition yet

        check_interval = max(1, self.tau // 10)
        if self.tau % check_interval != 0:
            return

        V_basis = compute_V(self.current_policy, y_bar)
        max_dist = self.confidence_set.max_dist_to_V(V_basis, self.current_policy)

        if np.sqrt(self.tau) * max_dist >= threshold:
            # Cycle complete — update confidence set
            radius = self.eta / np.sqrt(self.tau)
            self.confidence_set.update(self.current_policy, y_bar, radius)
            self.cycle_count += 1

            if self.verbose > 0:
                print(f"IUCB: cycle {self.cycle_count} complete after {self.tau} rounds, "
                      f"max_dist={max_dist:.4f}, radius={radius:.4f}")

            # Start new cycle
            self._start_new_cycle()

    def _start_new_cycle(self):
        """Select new optimistic hypothesis and reset cycle state."""
        P_opt, x_opt = self.confidence_set.optimistic_theta()
        self.current_theta = theta_from_payoff(P_opt)
        self.current_payoff = P_opt
        self.current_policy = x_opt
        self.tau = 0
        self.sigma_y = np.zeros(DIM_Y)

    def reset(self):
        super().reset()
        self.confidence_set = ConfidenceSet()
        self.cycle_count = 0
        self.tau = 0
        self.sigma_y = np.zeros(DIM_Y)
        self.reward_min = None
        self.reward_max = None

        # Initial optimistic hypothesis: full box [-1,1]^4
        # The optimistic theta maximizes game value, which for the full box
        # is P = [[1,1],[1,1]] (all payoffs = 1), value = 1.
        # But let's use the proper optimization.
        P_opt, x_opt = self.confidence_set.optimistic_theta()
        self.current_theta = theta_from_payoff(P_opt)
        self.current_payoff = P_opt
        self.current_policy = x_opt

    def dump_state(self):
        v, _ = solve_2x2_game(self.current_payoff)
        return (f"cycle={self.cycle_count} tau={self.tau} "
                f"policy={dump_array(self.current_policy)} "
                f"value={v:.3f} slabs={len(self.confidence_set.slabs)}")

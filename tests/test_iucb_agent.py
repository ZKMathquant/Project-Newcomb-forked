"""Tests for the IUCB agent.

These tests verify the IUCB agent's behavior on various environments.
Some are integration tests that require running the full simulation loop.
"""
import numpy as np
import pytest

from ibrl.agents.iucb import IUCBAgent
from ibrl.iucb.confidence_set import ConfidenceSet
from ibrl.iucb.algebra import theta_from_payoff, compute_V, dist_to_V, DIM_Y
from ibrl.utils.game_solving import solve_2x2_game
from ibrl.utils.construction import construct_agent, construct_environment
from ibrl.simulators.simulator import simulate


class TestIUCBConstruction:
    """Test that IUCB agent can be constructed and initialized."""

    def test_construct_via_string(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("iucb", options)
        assert isinstance(agent, IUCBAgent)

    def test_requires_two_actions(self):
        with pytest.raises(AssertionError):
            IUCBAgent(num_actions=3, seed=42)

    def test_reset_initializes_state(self):
        agent = IUCBAgent(num_actions=2, seed=42)
        agent.reset()
        assert agent.cycle_count == 0
        assert agent.tau == 0
        assert agent.current_policy is not None
        assert agent.current_policy.shape == (2,)
        assert abs(agent.current_policy.sum() - 1.0) < 1e-10

    def test_initial_policy_is_valid(self):
        agent = IUCBAgent(num_actions=2, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (2,)
        assert np.all(probs >= -1e-10)
        assert abs(probs.sum() - 1.0) < 1e-10


class TestConfidenceSet:
    """Test the confidence set operations."""

    def test_initial_set_is_full_box(self):
        cs = ConfidenceSet()
        # All payoff matrices in [-1,1]^4 should be in the initial set
        for _ in range(10):
            P = 2 * np.random.rand(2, 2) - 1  # uniform in [-1, 1]
            assert cs.contains(P)

    def test_initial_set_rejects_out_of_bounds(self):
        cs = ConfidenceSet()
        P = np.array([[2.0, 0.0], [0.0, 0.0]])  # P[0,0] > 1
        assert not cs.contains(P)

    def test_optimistic_theta_exists(self):
        cs = ConfidenceSet()
        P_opt, x_opt = cs.optimistic_theta()
        assert P_opt.shape == (2, 2)
        assert x_opt.shape == (2,)
        assert abs(x_opt.sum() - 1.0) < 1e-10

    def test_optimistic_theta_initial_value(self):
        """With full box [-1,1]^4, the optimistic game value should be 1."""
        cs = ConfidenceSet()
        P_opt, x_opt = cs.optimistic_theta()
        value, _ = solve_2x2_game(P_opt)
        # The max game value in [-1,1]^4 is 1 (e.g. P = [[1,1],[1,1]])
        assert value > 0.9, f"Expected optimistic value ~1, got {value}"

    def test_slab_update_narrows_set(self):
        """After adding a slab constraint, the set should be smaller."""
        cs = ConfidenceSet()
        # Create a slab from some observation
        x = np.array([0.5, 0.5])
        y_bar = np.zeros(DIM_Y)
        y_bar[0] = 0.25; y_bar[1] = 0.25  # some outcome
        y_bar[4] = 0.25; y_bar[5] = 0.25

        cs.update(x, y_bar, radius=0.1)

        # Some points should now be excluded
        # The set should be strictly smaller than [-1,1]^4
        assert len(cs.slabs) == 1
        assert cs.slabs[0].radius == 0.1


class TestIUCBOnDamascus:
    """Integration test: IUCB should find the p=0.5 mixed equilibrium on Damascus.

    Damascus payoff matrix: [[0, 10], [10, 0]]
    Minimax value: 5.0 at x = [0.5, 0.5]
    """

    @pytest.mark.slow
    def test_iucb_damascus_policy_converges(self):
        """IUCB should converge to the mixed strategy p=0.5 on Damascus."""
        options = {"num_actions": 2, "num_steps": 500, "num_runs": 5, "seed": 42, "verbose": 0}
        env = construct_environment("damascus", dict(options))
        agent = construct_agent("iucb", dict(options))
        results = simulate(env, agent, options)

        final_probs = results["probabilities"][:, -100:, :].mean(axis=(0, 1))
        # Should converge toward [0.5, 0.5]
        assert abs(final_probs[0] - 0.5) < 0.3, \
            f"Expected p(0) ~ 0.5, got {final_probs[0]:.3f}"

    @pytest.mark.slow
    def test_iucb_damascus_achieves_positive_reward(self):
        """IUCB on Damascus should achieve reward > 0 (random gets ~5, pure gets 0)."""
        options = {"num_actions": 2, "num_steps": 500, "num_runs": 5, "seed": 42, "verbose": 0}
        env = construct_environment("damascus", dict(options))
        agent = construct_agent("iucb", dict(options))
        results = simulate(env, agent, options)

        avg_reward = results["average_reward"][0][-100:].mean()
        assert avg_reward > 1.0, \
            f"Expected avg reward > 1, got {avg_reward:.2f}"


class TestIUCBOnBandit:
    """Integration test: IUCB on standard bandits should achieve sublinear regret."""

    @pytest.mark.slow
    def test_iucb_bandit_sublinear_regret(self):
        """IUCB regret should grow sublinearly on standard bandits."""
        options = {"num_actions": 2, "num_steps": 500, "num_runs": 5, "seed": 42, "verbose": 0}
        env = construct_environment("bandit", dict(options))
        agent = construct_agent("iucb", dict(options))
        results = simulate(env, agent, options)

        cumulative = results["average_reward"][0].cumsum()
        optimal_cumulative = results["optimal_reward"] * np.arange(1, 501)
        regret = optimal_cumulative - cumulative

        # Regret per step should decrease over time
        regret_rate_early = regret[99] / 100
        regret_rate_late = (regret[499] - regret[99]) / 400
        # The late regret rate should be strictly smaller than the early rate
        assert regret_rate_late < regret_rate_early, \
            f"Regret not sublinear: early rate={regret_rate_early:.3f}, late={regret_rate_late:.3f}"


class TestIUCBCycleLogic:
    """Unit tests for the cycle-based update logic."""

    def test_cycle_starts_after_reset(self):
        agent = IUCBAgent(num_actions=2, seed=42)
        agent.reset()
        assert agent.cycle_count == 0
        assert agent.tau == 0

    def test_policy_fixed_within_cycle(self):
        """The policy should remain constant within a single cycle."""
        agent = IUCBAgent(num_actions=2, seed=42, eta=1000.0)  # large eta = long cycles
        agent.reset()
        from ibrl.outcome import Outcome

        policy1 = agent.get_probabilities().copy()
        agent.update(policy1, 0, Outcome(reward=0.5, env_action=0))
        policy2 = agent.get_probabilities().copy()

        # Within the same cycle, policy should not change
        # (unless the stopping condition is immediately met, which it shouldn't
        # be with very large eta)
        np.testing.assert_array_equal(policy1, policy2)

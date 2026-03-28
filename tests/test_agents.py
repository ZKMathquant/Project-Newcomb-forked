import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent


class TestQLearningAgent:
    def test_initialization(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions
        assert agent.seed == seed

    def test_reset(self, q_learning_agent):
        q_learning_agent.reset()
        assert q_learning_agent.step == 1
        assert q_learning_agent.q.shape == (q_learning_agent.num_actions,)
        assert np.allclose(q_learning_agent.q, 0)

    def test_get_probabilities(self, q_learning_agent):
        probs = q_learning_agent.get_probabilities()
        assert probs.shape == (q_learning_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_update(self, q_learning_agent):
        probs = q_learning_agent.get_probabilities()
        action = 0
        reward = 1.0
        q_learning_agent.update(probs, action, reward)
        assert q_learning_agent.step == 2
        assert q_learning_agent.q[action] > 0

    def test_learning_rate_none(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, learning_rate=None, seed=seed)
        agent.reset()
        assert agent.learning_rate is None
        assert hasattr(agent, 'counts')

    def test_agent_rejects_invalid_num_actions(self):
        """Agent should reject invalid num_actions"""
        with pytest.raises(AssertionError):
            QLearningAgent(num_actions=1, seed=42)

    def test_probability_always_normalized(self, q_learning_agent):
        """Probabilities must always sum to 1"""
        for _ in range(10):
            probs = q_learning_agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0), f"Probs sum to {probs.sum()}"

    def test_q_values_remain_finite(self):
        """Q-values should never become NaN or Inf"""
        agent = QLearningAgent(num_actions=2, seed=42)
        agent.reset()
        for _ in range(100):
            probs = agent.get_probabilities()
            agent.update(probs, 0, np.random.randn() * 100)
        assert np.all(np.isfinite(agent.q)), "Q-values became non-finite"


class TestBayesianAgent:
    def test_initialization(self, num_actions, seed):
        agent = BayesianAgent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions

    def test_reset(self, bayesian_agent):
        bayesian_agent.reset()
        assert bayesian_agent.values.shape == (bayesian_agent.num_actions,)
        assert bayesian_agent.precision.shape == (bayesian_agent.num_actions,)
        assert np.allclose(bayesian_agent.values, 0)

    def test_get_probabilities(self, bayesian_agent):
        probs = bayesian_agent.get_probabilities()
        assert probs.shape == (bayesian_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_update_increases_precision(self, bayesian_agent):
        initial_precision = bayesian_agent.precision.copy()
        probs = bayesian_agent.get_probabilities()
        bayesian_agent.update(probs, 0, 1.0)
        assert bayesian_agent.precision[0] > initial_precision[0]

    def test_values_remain_finite(self, bayesian_agent):
        """Bayesian values should never become NaN or Inf"""
        for _ in range(100):
            probs = bayesian_agent.get_probabilities()
            bayesian_agent.update(probs, 0, np.random.randn() * 100)
        assert np.all(np.isfinite(bayesian_agent.values))


class TestEXP3Agent:
    def test_initialization(self, num_actions, seed):
        agent = EXP3Agent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions
        assert agent.gamma == 0.1

    def test_reset(self, exp3_agent):
        exp3_agent.reset()
        assert exp3_agent.log_weights.shape == (exp3_agent.num_actions,)
        assert np.allclose(exp3_agent.log_weights, 0)

    def test_get_probabilities(self, exp3_agent):
        probs = exp3_agent.get_probabilities()
        assert probs.shape == (exp3_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_update_changes_weights(self, exp3_agent):
        initial_weights = exp3_agent.log_weights.copy()
        probs = exp3_agent.get_probabilities()
        exp3_agent.update(probs, 0, 1.0)
        assert not np.allclose(exp3_agent.log_weights, initial_weights)

    def test_probabilities_always_valid(self, exp3_agent):
        """Probabilities must always be valid"""
        for _ in range(50):
            probs = exp3_agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0)
            assert np.all(probs >= 0)
            assert np.all(np.isfinite(probs))
            exp3_agent.update(probs, 0, 1.0)

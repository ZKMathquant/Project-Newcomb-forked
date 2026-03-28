import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.utils import dump_array


class TestBaseGreedyAgent:
    """Test BaseGreedyAgent functionality through QLearningAgent"""

    def test_epsilon_greedy_policy(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, epsilon=0.1, seed=seed)
        agent.reset()
        agent.q = np.array([1.0, 0.0])
        probs = agent.build_epsilon_greedy_policy(agent.q)
        assert np.isclose(probs.sum(), 1.0)
        assert probs[0] > probs[1]

    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy with high epsilon"""
        agent = QLearningAgent(num_actions=2, epsilon=0.9, seed=42)
        agent.reset()
        agent.q = np.array([10.0, 0.0])
        probs = agent.build_epsilon_greedy_policy(agent.q)
        assert probs[1] > 0.1
        assert np.isclose(probs.sum(), 1.0)

    def test_epsilon_greedy_exploitation(self):
        """Test epsilon-greedy with low epsilon favors best action"""
        agent = QLearningAgent(num_actions=2, epsilon=0.01, seed=42)
        agent.reset()
        agent.q = np.array([10.0, 0.0])
        probs = agent.build_epsilon_greedy_policy(agent.q)
        assert probs[0] > 0.9
        assert np.isclose(probs.sum(), 1.0)

    def test_softmax_policy(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, temperature=1.0, seed=seed)
        agent.reset()
        agent.q = np.array([1.0, 0.0])
        probs = agent.build_softmax_policy(agent.q)
        assert np.isclose(probs.sum(), 1.0)
        assert probs[0] > probs[1]

    def test_softmax_temperature_effect(self):
        """Test softmax with different temperatures"""
        agent = QLearningAgent(num_actions=2, temperature=0.1, seed=42)
        agent.reset()
        agent.q = np.array([1.0, 0.0])
        probs_cold = agent.build_softmax_policy(agent.q)
        
        agent.temperature = 10.0
        probs_hot = agent.build_softmax_policy(agent.q)
        
        assert probs_cold[0] > probs_hot[0]
        assert np.isclose(probs_cold.sum(), 1.0)
        assert np.isclose(probs_hot.sum(), 1.0)

    def test_softmax_uniform_at_high_temperature(self):
        """Test softmax approaches uniform distribution at high temperature"""
        agent = QLearningAgent(num_actions=2, temperature=100.0, seed=42)
        agent.reset()
        agent.q = np.array([10.0, 0.0])
        probs = agent.build_softmax_policy(agent.q)
        assert np.allclose(probs, [0.5, 0.5], atol=0.1)

    def test_exponential_decay(self, num_actions, seed):
        agent = QLearningAgent(
            num_actions=num_actions,
            epsilon=(1.0, 0.5, 0.01),
            decay_type=0,
            seed=seed
        )
        agent.reset()
        eps1 = agent.parse_parameter(agent.epsilon)
        agent.step = 10
        eps2 = agent.parse_parameter(agent.epsilon)
        assert eps2 < eps1

    def test_exponential_decay_respects_minimum(self):
        """Test exponential decay doesn't go below minimum"""
        agent = QLearningAgent(
            num_actions=2,
            epsilon=(1.0, 0.5, 0.01),
            decay_type=0,
            seed=42
        )
        agent.reset()
        agent.step = 100
        eps = agent.parse_parameter(agent.epsilon)
        assert eps >= 0.01

    def test_linear_decay(self, num_actions, seed):
        agent = QLearningAgent(
            num_actions=num_actions,
            epsilon=(1.0, 500, 0.01),
            decay_type=1,
            seed=seed
        )
        agent.reset()
        eps1 = agent.parse_parameter(agent.epsilon)
        agent.step = 250
        eps2 = agent.parse_parameter(agent.epsilon)
        assert eps2 < eps1

    def test_linear_decay_reaches_minimum(self):
        """Test linear decay reaches minimum after decay period"""
        agent = QLearningAgent(
            num_actions=2,
            epsilon=(1.0, 100, 0.01),
            decay_type=1,
            seed=42
        )
        agent.reset()
        agent.step = 200
        eps = agent.parse_parameter(agent.epsilon)
        assert eps == 0.01

    def test_parse_parameter_fixed_value(self):
        """Test parse_parameter with fixed float value"""
        agent = QLearningAgent(num_actions=2, epsilon=0.5, seed=42)
        agent.reset()
        eps = agent.parse_parameter(agent.epsilon)
        assert eps == 0.5


class TestQLearningAgentExtended:
    """Extended tests for QLearningAgent"""

    def test_sample_average_mode(self):
        """Test Q-learning with sample averaging"""
        agent = QLearningAgent(num_actions=2, learning_rate=None, seed=42)
        agent.reset()
        
        probs = agent.get_probabilities()
        agent.update(probs, 0, 1.0)
        assert agent.counts[0] == 1
        assert np.isclose(agent.q[0], 1.0)
        
        agent.update(probs, 0, 3.0)
        assert agent.counts[0] == 2
        assert np.isclose(agent.q[0], 2.0)

    def test_sample_average_multiple_actions(self):
        """Test sample averaging across different actions"""
        agent = QLearningAgent(num_actions=2, learning_rate=None, seed=42)
        agent.reset()
        
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 2.0)
        agent.update(probs, 0, 4.0)
        agent.update(probs, 1, 1.0)
        
        assert np.isclose(agent.q[0], 3.0)
        assert np.isclose(agent.q[1], 1.0)
        assert agent.counts[0] == 2
        assert agent.counts[1] == 1

    def test_learning_rate_mode(self):
        """Test Q-learning with fixed learning rate"""
        agent = QLearningAgent(num_actions=2, learning_rate=0.1, seed=42)
        agent.reset()
        
        probs = agent.get_probabilities()
        initial_q = agent.q[0]
        
        agent.update(probs, 0, 1.0)
        
        expected_q = initial_q + 0.1 * (1.0 - initial_q)
        assert np.isclose(agent.q[0], expected_q)


class TestBayesianAgentExtended:
    """Extended tests for BayesianAgent"""

    def test_precision_increases_with_updates(self):
        """Test that precision increases with each update"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        
        initial_precision = agent.precision[0]
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 1.0)
        assert agent.precision[0] > initial_precision
        
        agent.update(probs, 0, 1.0)
        assert agent.precision[0] > initial_precision + 1

    def test_value_converges_to_reward(self):
        """Test that values converge to consistent rewards"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        
        probs = agent.get_probabilities()
        
        for _ in range(10):
            agent.update(probs, 0, 5.0)
        
        assert np.isclose(agent.values[0], 5.0, atol=0.5)


class TestEXP3AgentExtended:
    """Extended tests for EXP3Agent"""

    def test_weights_change_with_reward(self):
        """Test that weights change based on rewards"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        
        initial_weights = agent.log_weights.copy()
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 1.0)
        
        assert not np.allclose(agent.log_weights, initial_weights)

    def test_probabilities_sum_to_one(self):
        """Test that probabilities always sum to 1"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        
        for _ in range(10):
            probs = agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0)
            agent.update(probs, 0, 1.0)


class TestDebugUtils:
    """Test debug utility functions"""

    def test_dump_array(self):
        arr = np.array([0.5, 0.3, 0.2])
        result = dump_array(arr)
        assert "[" in result
        assert "]" in result
        assert "0.50" in result

    def test_dump_array_single_value(self):
        arr = np.array([1.0])
        result = dump_array(arr)
        assert "1.00" in result

    def test_dump_array_negative_values(self):
        arr = np.array([-0.5, 0.5])
        result = dump_array(arr)
        assert "-0.50" in result

import numpy as np
import pytest
from ibrl.environments import (
    BanditEnvironment,
    NewcombEnvironment,
    DeathInDamascusEnvironment,
    CoordinationGameEnvironment,
    PolicyDependentBanditEnvironment
)


class TestBanditEnvironment:
    def test_initialization(self, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        assert env.num_actions == num_actions

    def test_reset(self, bandit_env):
        bandit_env.reset()
        assert bandit_env.rewards.shape == (bandit_env.num_actions,)

    def test_interact(self, bandit_env):
        reward = bandit_env.interact(0)
        assert isinstance(reward, (float, np.floating))

    def test_get_optimal_reward(self, bandit_env):
        optimal = bandit_env.get_optimal_reward()
        assert isinstance(optimal, (float, np.floating))
        assert optimal == bandit_env.rewards.max()


class TestNewcombEnvironment:
    def test_initialization(self, seed):
        env = NewcombEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2
        assert env.reward_table.shape == (2, 2)

    def test_reward_table_structure(self, newcomb_env):
        assert newcomb_env.reward_table[0, 0] == 10  # boxB
        assert newcomb_env.reward_table[0, 1] == 15  # boxB + boxA
        assert newcomb_env.reward_table[1, 0] == 0
        assert newcomb_env.reward_table[1, 1] == 5   # boxA

    def test_predict_sets_rewards(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        newcomb_env.predict(probs)
        assert hasattr(newcomb_env, 'rewards')
        assert newcomb_env.rewards.shape == (2,)

    def test_interact(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        newcomb_env.predict(probs)
        reward = newcomb_env.interact(0)
        assert isinstance(reward, (int, float, np.integer, np.floating))
        assert reward in [0, 5, 10, 15]

    def test_get_optimal_reward(self, newcomb_env):
        optimal = newcomb_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert optimal >= 0


class TestDeathInDamascusEnvironment:
    def test_initialization(self, seed):
        env = DeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, damascus_env):
        assert damascus_env.reward_table[0, 0] == 0   # death in Damascus
        assert damascus_env.reward_table[0, 1] == 10  # life
        assert damascus_env.reward_table[1, 0] == 10  # life
        assert damascus_env.reward_table[1, 1] == 0   # death in Damascus

    def test_get_optimal_reward(self, damascus_env):
        optimal = damascus_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert 0 <= optimal <= 10


class TestCoordinationGameEnvironment:
    def test_initialization(self, seed):
        env = CoordinationGameEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, seed):
        env = CoordinationGameEnvironment(num_actions=2, rewardA=2, rewardB=1, seed=seed)
        assert env.reward_table[0, 0] == 2
        assert env.reward_table[1, 1] == 1

    def test_custom_rewards(self, seed):
        env = CoordinationGameEnvironment(num_actions=2, rewardA=5, rewardB=3, seed=seed)
        assert env.reward_table[0, 0] == 5
        assert env.reward_table[1, 1] == 3


class TestPolicyDependentBanditEnvironment:
    def test_initialization(self, seed):
        env = PolicyDependentBanditEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reset_generates_random_table(self, seed):
        env = PolicyDependentBanditEnvironment(num_actions=2, seed=seed)
        env.reset()
        assert env.reward_table.shape == (2, 2)

    def test_random_reward_table(self):
        """Test that reward table is randomized with different seeds"""
        env1 = PolicyDependentBanditEnvironment(num_actions=2, seed=42)
        env1.reset()
        
        env2 = PolicyDependentBanditEnvironment(num_actions=2, seed=43)
        env2.reset()
        
        # Different seeds = different tables
        assert not np.allclose(env1.reward_table, env2.reward_table)

    def test_predict_and_interact(self, seed):
        env = PolicyDependentBanditEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(0)
        assert isinstance(reward, (int, float, np.floating))

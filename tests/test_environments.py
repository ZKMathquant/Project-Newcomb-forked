import numpy as np
import pytest
from ibrl.environments import (
    BanditEnvironment,
    NewcombEnvironment,
    DeathInDamascusEnvironment,
    CoordinationGameEnvironment,
    PolicyDependentBanditEnvironment,
    AsymmetricDeathInDamascusEnvironment,
    SwitchingAdversaryEnvironment,
    MatchEnvironment,
    ReverseTailsEnvironment
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

    def test_reward_is_always_finite(self):
        """Rewards should never be NaN or Inf"""
        env = BanditEnvironment(num_actions=2, seed=42)
        for _ in range(100):
            env.reset()
            reward = env.interact(0)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_environment_rejects_invalid_actions(self):
        """Environment should reject actions outside range"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        with pytest.raises((IndexError, AssertionError)):
            env.interact(5)


class TestNewcombEnvironment:
    def test_initialization(self, seed):
        env = NewcombEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2
        assert env.reward_table.shape == (2, 2)

    def test_reward_table_structure(self, newcomb_env):
        assert newcomb_env.reward_table[0, 0] == 10
        assert newcomb_env.reward_table[0, 1] == 15
        assert newcomb_env.reward_table[1, 0] == 0
        assert newcomb_env.reward_table[1, 1] == 5

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
        assert damascus_env.reward_table[0, 0] == 0
        assert damascus_env.reward_table[0, 1] == 10
        assert damascus_env.reward_table[1, 0] == 10
        assert damascus_env.reward_table[1, 1] == 0

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
        
        assert not np.allclose(env1.reward_table, env2.reward_table)

    def test_predict_and_interact(self, seed):
        env = PolicyDependentBanditEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(0)
        assert isinstance(reward, (int, float, np.floating))


class TestAsymmetricDeathInDamascusEnvironment:
    def test_initialization(self, seed):
        env = AsymmetricDeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_default_rewards(self, seed):
        env = AsymmetricDeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 0
        assert env.reward_table[1, 1] == 5
        assert env.reward_table[0, 1] == 10
        assert env.reward_table[1, 0] == 10

    def test_custom_rewards(self, seed):
        env = AsymmetricDeathInDamascusEnvironment(
            num_actions=2,
            death_in_damascus=1,
            death_in_aleppo=2,
            life=3,
            seed=seed
        )
        assert env.reward_table[0, 0] == 1
        assert env.reward_table[1, 1] == 2
        assert env.reward_table[0, 1] == 3
        assert env.reward_table[1, 0] == 3

    def test_asymmetry_in_rewards(self, seed):
        env = AsymmetricDeathInDamascusEnvironment(
            num_actions=2,
            death_in_damascus=0,
            death_in_aleppo=5,
            life=10,
            seed=seed
        )
        assert env.reward_table[0, 0] != env.reward_table[1, 1]
        assert env.reward_table[0, 0] == 0
        assert env.reward_table[1, 1] == 5


class TestSwitchingAdversaryEnvironment:
    def test_initialization(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        assert env.num_actions == 2

    def test_switch_at_parameter(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=50, seed=seed)
        assert env.switch_at == 50

    def test_default_switch_at(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        assert env.switch_at == 50

    def test_reset(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        env.reset()
        assert env.step == 0
        assert env.values[0] == 1.0
        assert env.values[1] == 0.0

    def test_interact_before_switch(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=50, seed=seed)
        env.reset()
        reward = env.interact(0)
        assert isinstance(reward, (int, float, np.floating))
        assert env.step == 1

    def test_interact_after_switch(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=2, seed=seed)
        env.reset()
        
        env.interact(0)
        env.interact(0)
        env.interact(0)
        assert env.values[-1] == 1.0
        assert env.values[0] == 0.0

    def test_get_optimal_reward(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        optimal = env.get_optimal_reward()
        assert optimal == 1.0

    def test_step_increments(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        env.reset()
        assert env.step == 0
        
        env.interact(0)
        assert env.step == 1
        
        env.interact(1)
        assert env.step == 2


class TestMatchEnvironment:
    def test_initialization(self, seed):
        env = MatchEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, seed):
        env = MatchEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 1.0
        assert env.reward_table[0, 1] == 0.0
        assert env.reward_table[1, 0] == 0.0
        assert env.reward_table[1, 1] == 1.0

    def test_predict_and_interact(self, seed):
        env = MatchEnvironment(num_actions=2, seed=seed)
        env.reset()
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(0)
        
        assert reward == 1.0

    def test_mismatch_reward(self, seed):
        """Test mismatch gives 0 reward"""
        env = MatchEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(1)
        
        assert reward == 0.0


class TestReverseTailsEnvironment:
    def test_initialization(self, seed):
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, seed):
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 0.0
        assert env.reward_table[0, 1] == 1.0
        assert env.reward_table[1, 0] == 0.5
        assert env.reward_table[1, 1] == 0.5

    def test_heads_mismatch_reward(self, seed):
        """Test mismatch on heads gives 1.0"""
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(1)
        
        assert reward == 1.0

    def test_tails_always_half(self, seed):
        """Test tails always gives 0.5 regardless of action"""
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([0.0, 1.0])
        env.predict(probs)
        
        reward_action0 = env.interact(0)
        assert reward_action0 == 0.5
        
        env.reset()
        env.predict(probs)
        reward_action1 = env.interact(1)
        assert reward_action1 == 0.5

    def test_asymmetric_behavior(self, seed):
        """Test that heads and tails have different reward structures"""
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        
        assert env.reward_table[0, 1] > env.reward_table[0, 0]
        assert env.reward_table[1, 0] == env.reward_table[1, 1]

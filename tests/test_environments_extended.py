import numpy as np
import pytest
from ibrl.environments import (
    AsymmetricDeathInDamascusEnvironment,
    SwitchingAdversaryEnvironment,
    MatchEnvironment,
    ReverseTailsEnvironment
)


class TestAsymmetricDeathInDamascusEnvironment:
    """Extended tests for AsymmetricDeathInDamascusEnvironment"""

    def test_initialization(self, seed):
        env = AsymmetricDeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_default_rewards(self, seed):
        """Test default reward values"""
        env = AsymmetricDeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 0   # death in Damascus
        assert env.reward_table[1, 1] == 5   # death in Aleppo
        assert env.reward_table[0, 1] == 10  # life
        assert env.reward_table[1, 0] == 10  # life

    def test_custom_rewards(self, seed):
        """Test custom reward values"""
        env = AsymmetricDeathInDamascusEnvironment(
            num_actions=2,
            death_in_damascus=1,
            death_in_aleppo=2,
            life=3,
            seed=seed
        )
        assert env.reward_table[0, 0] == 1  # death in Damascus
        assert env.reward_table[1, 1] == 2  # death in Aleppo
        assert env.reward_table[0, 1] == 3  # life
        assert env.reward_table[1, 0] == 3  # life

    def test_asymmetry_in_rewards(self, seed):
        """Test that Damascus and Aleppo have different death rewards"""
        env = AsymmetricDeathInDamascusEnvironment(
            num_actions=2,
            death_in_damascus=0,
            death_in_aleppo=5,
            life=10,
            seed=seed
        )
        # Verify asymmetry
        assert env.reward_table[0, 0] != env.reward_table[1, 1]
        assert env.reward_table[0, 0] == 0
        assert env.reward_table[1, 1] == 5


class TestSwitchingAdversaryEnvironment:
    """Extended tests for SwitchingAdversaryEnvironment"""

    def test_initialization(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        assert env.num_actions == 2

    def test_switch_at_parameter(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=50, seed=seed)
        assert env.switch_at == 50

    def test_default_switch_at(self, seed):
        """Test default switch_at is num_steps // 2"""
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
        """Test that reward switches to last arm after switch point"""
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=2, seed=seed)
        env.reset()
        
        # Before switch
        env.interact(0)
        env.interact(0)
        
        # After switch - reward should be on last arm
        env.interact(0)
        assert env.values[-1] == 1.0
        assert env.values[0] == 0.0

    def test_get_optimal_reward(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        optimal = env.get_optimal_reward()
        assert optimal == 1.0

    def test_step_increments(self, seed):
        """Test that step counter increments correctly"""
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        env.reset()
        assert env.step == 0
        
        env.interact(0)
        assert env.step == 1
        
        env.interact(1)
        assert env.step == 2


class TestMatchEnvironment:
    """Tests for MatchEnvironment (coin tossing)"""

    def test_initialization(self, seed):
        env = MatchEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, seed):
        """Test reward table for match environment"""
        env = MatchEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 1.0  # match on H
        assert env.reward_table[0, 1] == 0.0  # mismatch on H
        assert env.reward_table[1, 0] == 0.0  # mismatch on T
        assert env.reward_table[1, 1] == 1.0  # match on T

    def test_predict_and_interact(self, seed):
        """Test predict and interact flow"""
        env = MatchEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(0)
        
        # Should get reward 1.0 (match)
        assert reward == 1.0

    def test_mismatch_reward(self, seed):
        """Test mismatch gives 0 reward"""
        env = MatchEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(1)
        
        # Should get reward 0.0 (mismatch)
        assert reward == 0.0


class TestReverseTailsEnvironment:
    """Tests for ReverseTailsEnvironment (coin tossing variant)"""

    def test_initialization(self, seed):
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, seed):
        """Test reward table for reverse tails environment"""
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        assert env.reward_table[0, 0] == 0.0   # match on H
        assert env.reward_table[0, 1] == 1.0   # mismatch on H
        assert env.reward_table[1, 0] == 0.5   # T always 0.5
        assert env.reward_table[1, 1] == 0.5   # T always 0.5

    def test_heads_mismatch_reward(self, seed):
        """Test mismatch on heads gives 1.0"""
        env = ReverseTailsEnvironment(num_actions=2, seed=seed)
        env.reset()
        
        probs = np.array([1.0, 0.0])
        env.predict(probs)
        reward = env.interact(1)
        
        # Should get reward 1.0 (mismatch on H)
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
        
        # Heads: mismatch is good (1.0), match is bad (0.0)
        assert env.reward_table[0, 1] > env.reward_table[0, 0]
        
        # Tails: always 0.5
        assert env.reward_table[1, 0] == env.reward_table[1, 1]

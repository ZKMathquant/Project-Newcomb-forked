"""Tests for infrabayesian module — faithful port from coin-learning.ipynb."""
import numpy as np
import pytest

from ibrl.infrabayesian import AMeasure, Infradistribution, match, glue, Coin


# ── Reward functions (same as notebook) ──────────────────────────────────────

def reward_zero(history):
    return 0

def reward_one(history):
    return 1

def reward_arbitrary(history):
    """Deterministic pseudo-random reward. The notebook uses hash() but that's
    not stable across Python sessions, so we use a fixed mapping."""
    # Map each 2-flip history to a fixed value
    mapping = {
        (Coin.H, Coin.H): 0.123,
        (Coin.H, Coin.T): 0.456,
        (Coin.T, Coin.H): 0.789,
        (Coin.T, Coin.T): 0.012,
    }
    return mapping.get(history, 0.5)


# ── History space (same as notebook) ─────────────────────────────────────────

X = [
    (Coin.H, Coin.H),
    (Coin.H, Coin.T),
    (Coin.T, Coin.H),
    (Coin.T, Coin.T),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def a_measure_from_hypothesis(p):
    """Build an a-measure for a coin with P(H) = p."""
    p_obs = [p, 1 - p]
    mu = np.array([
        p_obs[x[0].value] * p_obs[x[1].value] for x in X
    ])
    return AMeasure(mu, history_space=X)


def make_hypothesis_measures():
    """Three hypotheses: p = 0.3, 0.5, 0.7."""
    mA = a_measure_from_hypothesis(0.3)
    mB = a_measure_from_hypothesis(0.5)
    mC = a_measure_from_hypothesis(0.7)
    return mA, mB, mC


# ── match() tests ────────────────────────────────────────────────────────────

class TestMatch:
    def test_exact_match(self):
        assert match((Coin.H, Coin.T), (Coin.H, Coin.T)) is True

    def test_exact_mismatch(self):
        assert match((Coin.H, Coin.T), (Coin.H, Coin.H)) is False

    def test_wildcard(self):
        assert match((Coin.H, None), (Coin.H, Coin.H)) is True
        assert match((Coin.H, None), (Coin.H, Coin.T)) is True
        assert match((Coin.H, None), (Coin.T, Coin.H)) is False

    def test_list_pattern(self):
        pattern = [(Coin.H, Coin.H), (Coin.T, Coin.T)]
        assert match(pattern, (Coin.H, Coin.H)) is True
        assert match(pattern, (Coin.T, Coin.T)) is True
        assert match(pattern, (Coin.H, Coin.T)) is False


# ── AMeasure tests ───────────────────────────────────────────────────────────

class TestAMeasure:
    def test_hypothesis_measures(self):
        """Reproduce notebook a-measure values for each hypothesis."""
        mA = a_measure_from_hypothesis(0.3)
        # p=0.3: HH=0.09, HT=0.21, TH=0.21, TT=0.49
        np.testing.assert_allclose(mA.measure, [0.09, 0.21, 0.21, 0.49], atol=1e-10)
        assert mA.scale == pytest.approx(1.0, abs=1e-10)
        assert mA.offset == pytest.approx(0.0, abs=1e-10)

    def test_mixture(self):
        """Uniform mixture of 3 hypotheses: notebook shows [0.277, 0.223, 0.223, 0.277]."""
        mA, mB, mC = make_hypothesis_measures()
        mix = (mA + mB + mC) / 3
        np.testing.assert_allclose(
            mix.scale * mix.measure,
            [0.277, 0.223, 0.223, 0.277],
            atol=0.001,
        )

    def test_expected_value_of_constant_one(self):
        """α(1) should equal scale for a fresh a-measure (offset=0)."""
        mA = a_measure_from_hypothesis(0.5)
        assert mA.expected_value(reward_one) == pytest.approx(1.0, abs=1e-10)

    def test_expected_value_of_constant_zero(self):
        mA = a_measure_from_hypothesis(0.5)
        assert mA.expected_value(reward_zero) == pytest.approx(0.0, abs=1e-10)

    def test_addition(self):
        mA = a_measure_from_hypothesis(0.3)
        mB = a_measure_from_hypothesis(0.7)
        s = mA + mB
        # (mA + mB)(1) should be 2.0
        assert s.expected_value(reward_one) == pytest.approx(2.0, abs=1e-10)

    def test_division(self):
        mA = a_measure_from_hypothesis(0.5)
        halved = mA / 2
        assert halved.expected_value(reward_one) == pytest.approx(0.5, abs=1e-10)


# ── glue() tests ─────────────────────────────────────────────────────────────

class TestGlue:
    def test_glue_matches_pattern(self):
        event_H = (Coin.H, None)
        glued = glue(reward_one, event_H, reward_zero)
        # HH matches event_H -> reward_one -> 1
        assert glued((Coin.H, Coin.H)) == 1
        # HT matches event_H -> reward_one -> 1
        assert glued((Coin.H, Coin.T)) == 1
        # TH does not match -> reward_zero -> 0
        assert glued((Coin.T, Coin.H)) == 0
        # TT does not match -> reward_zero -> 0
        assert glued((Coin.T, Coin.T)) == 0


# ── Infradistribution: classical (non-KU) ───────────────────────────────────

class TestClassicalCoinFlip:
    """Reproduce the notebook's classical IB experiment."""

    def test_p_heads_first_flip(self):
        """P(H) = 0.500 under uniform prior with 3 hypotheses."""
        mA, mB, mC = make_hypothesis_measures()
        mix = (mA + mB + mC) / 3
        infradist = Infradistribution([mix])

        event_H = (Coin.H, None)
        p_h = infradist.probability(reward_arbitrary, event_H)
        assert p_h == pytest.approx(0.500, abs=0.001)

    def test_p_hh_given_h(self):
        """After observing H, P(H₂|H₁) = 0.553."""
        mA, mB, mC = make_hypothesis_measures()
        mix = (mA + mB + mC) / 3
        infradist = Infradistribution([mix])

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)

        infradist.update(reward_arbitrary, event_H)
        p_hh = infradist.probability(reward_arbitrary, event_HH)
        assert p_hh == pytest.approx(0.553, abs=0.001)

    def test_reward_function_independence_nonku(self):
        """For single-a-measure infradist, results are independent of reward function."""
        reward_funcs = [reward_zero, reward_one, reward_arbitrary]

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)

        probabilities = []
        for rf in reward_funcs:
            mA, mB, mC = make_hypothesis_measures()
            mix = (mA + mB + mC) / 3
            infradist = Infradistribution([mix])
            infradist.update(rf, event_H)
            p = infradist.probability(rf, event_HH)
            probabilities.append(p)

        # All should give 0.553
        for p in probabilities:
            assert p == pytest.approx(0.553, abs=0.001)


# ── Infradistribution: KU ───────────────────────────────────────────────────

class TestKUCoinFlip:
    """Reproduce the notebook's KU experiment outputs."""

    def test_ku_initial_probabilities(self):
        """KU initial P(H) depends on reward function."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        event_T = (Coin.T, None)

        # With reward_zero: P(H) = 0.300, P(T) = 0.300
        assert infradist.probability(reward_zero, event_H) == pytest.approx(0.300, abs=0.001)
        assert infradist.probability(reward_zero, event_T) == pytest.approx(0.300, abs=0.001)

        # With reward_one: P(H) = 0.700, P(T) = 0.700
        assert infradist.probability(reward_one, event_H) == pytest.approx(0.700, abs=0.001)
        assert infradist.probability(reward_one, event_T) == pytest.approx(0.700, abs=0.001)

    def test_ku_update_with_zero_reward(self):
        """KU update with reward_zero: P(HH)=0.300, P(HT)=0.700 regardless of reward func used to query."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)
        event_HT = (Coin.H, Coin.T)

        infradist.update(reward_zero, event_H)

        # Notebook: all reward functions give P(HH)=0.300, P(HT)=0.700
        for rf in [reward_zero, reward_one, reward_arbitrary]:
            assert infradist.probability(rf, event_HH) == pytest.approx(0.300, abs=0.001)
            assert infradist.probability(rf, event_HT) == pytest.approx(0.700, abs=0.001)

    def test_ku_update_with_one_reward(self):
        """KU update with reward_one: updated scales from notebook."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)

        infradist.update(reward_one, event_H)

        # Notebook output shows scales: 0.429, 0.714, 1.000
        assert infradist.measures[0].scale == pytest.approx(0.429, abs=0.001)
        assert infradist.measures[1].scale == pytest.approx(0.714, abs=0.001)
        assert infradist.measures[2].scale == pytest.approx(1.000, abs=0.001)

        # Notebook: offsets 0.571, 0.286, 0.000
        assert infradist.measures[0].offset == pytest.approx(0.571, abs=0.001)
        assert infradist.measures[1].offset == pytest.approx(0.286, abs=0.001)
        assert infradist.measures[2].offset == pytest.approx(0.000, abs=0.001)

    def test_ku_update_with_arbitrary_reward(self):
        """After KU update with a non-degenerate reward func, querying P with
        different reward functions should all agree (reward-function independence
        emerges when the update reward is non-degenerate)."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)
        event_HT = (Coin.H, Coin.T)

        infradist.update(reward_arbitrary, event_H)

        # All reward functions should give the same probabilities
        probs_hh = [infradist.probability(rf, event_HH) for rf in [reward_zero, reward_one, reward_arbitrary]]
        probs_ht = [infradist.probability(rf, event_HT) for rf in [reward_zero, reward_one, reward_arbitrary]]

        # Check consistency: all queries agree
        assert probs_hh[0] == pytest.approx(probs_hh[1], abs=0.001)
        assert probs_hh[0] == pytest.approx(probs_hh[2], abs=0.001)
        assert probs_ht[0] == pytest.approx(probs_ht[1], abs=0.001)
        assert probs_ht[0] == pytest.approx(probs_ht[2], abs=0.001)

        # P(HH) + P(HT) should sum to ~1 (they're the only outcomes given H)
        assert probs_hh[0] + probs_ht[0] == pytest.approx(1.0, abs=0.001)


# ── Phase 2: Bug fix tests ──────────────────────────────────────────────────

class TestOffsetBugFix:
    """The notebook's += in Infradistribution.update() double-counts offset b.
    After fix (= instead of +=), multi-step KU updates should remain valid."""

    def test_offset_assignment_not_accumulation(self):
        """After KU update with reward_one, do a second update on event_HH.
        With the bug (+=), the offset accumulates incorrectly.
        With the fix (=), alpha(1) should remain <= 1 for all measures."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        event_HH = (Coin.H, Coin.H)

        # First update gives nonzero offsets (0.571, 0.286, 0.000)
        infradist.update(reward_one, event_H)

        # Second update — the bug would double-count offsets here
        infradist.update(reward_one, event_HH)

        # After fix: all a-measures should be valid (alpha(1) <= 1 + epsilon)
        for m in infradist.measures:
            alpha_one = m.expected_value(reward_one)
            assert alpha_one <= 1.0 + 1e-6, f"alpha(1) = {alpha_one} > 1, offset bug likely present"

    def test_offset_is_fresh_not_accumulated(self):
        """After a KU update that produces nonzero offsets, verify the offset
        values are computed fresh (= formula), not accumulated from prior state."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        event_H = (Coin.H, None)
        infradist.update(reward_one, event_H)

        # Record offsets after first update
        offsets_after_first = [m.offset for m in infradist.measures]
        # Notebook: [0.571, 0.286, 0.000] — mA has highest offset
        assert offsets_after_first[0] > offsets_after_first[1] > offsets_after_first[2]
        assert offsets_after_first[2] == pytest.approx(0.0, abs=1e-6)  # mC offset is 0


class TestNormalizationValidation:
    """After KU update with degenerate reward (f=0), a-measures can become
    invalid (alpha(1) > 1). AMeasure.is_valid() should detect this."""

    def test_is_valid_on_fresh_measure(self):
        """A fresh a-measure should be valid."""
        m = a_measure_from_hypothesis(0.5)
        assert m.is_valid()

    def test_is_valid_after_normal_update(self):
        """After non-degenerate update, measures should be valid."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])
        infradist.update(reward_one, (Coin.H, None))
        for m in infradist.measures:
            assert m.is_valid()

    def test_update_with_zero_reward_warns(self):
        """KU update with reward_zero produces invalid a-measures. Should warn."""
        mA, mB, mC = make_hypothesis_measures()
        infradist = Infradistribution([mA, mB, mC])

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            infradist.update(reward_zero, (Coin.H, None))
            # Should produce a warning about invalid a-measures
            assert any("valid" in str(warning.message).lower() or
                       "normalization" in str(warning.message).lower()
                       for warning in w), \
                f"Expected normalization warning, got: {[str(x.message) for x in w]}"

# Implementation Plan: Simplified IUCB for Newcomb-Like Games

**Date**: 2026-03-21
**Context**: Replaces the thesis-faithful IUCB (Kosoy 2024) with a direct
per-cell confidence interval approach for K×K payoff matrix games.
**Predecessor**: `20260320_ibplan.md` (Sections 16-17 document why the
full IUCB implementation is impractical).

---

## 1. Motivation and Background

### 1.1 Why Replace the Full IUCB?

The full IUCB implementation (Kosoy 2024, Algorithm 4) faithfully follows
the thesis's general framework for linear imprecise bandits. Through
implementation and analysis, we discovered several fundamental practical
limitations:

1. **The formal regret guarantee requires ~20M rounds per cycle** even
   for 2×2 games. The theoretical constants (R=12, D_W=6, D_Z=10) make
   the prescribed eta ~200, giving a stopping threshold of ~4400.

2. **The heuristic eta that makes it practical abandons the guarantee.**
   We use eta = N^{1/4} / (2*(D_Z+1)), which is ~400x smaller than the
   theoretical prescription. No known regret bound applies.

3. **The representation is inflated.** The 2×2 payoff matrix has 4 free
   parameters, but the thesis embeds it in a 10D hypothesis space Z and
   6D constraint space W. The stopping threshold scales with D_Z+1=11
   instead of 4+1=5.

4. **The cycle structure imposes a minimum experiment length.** IUCB
   needs ~500 steps to complete enough cycles on 2×2 games. Simpler
   algorithms (UCB, EXP3) begin adapting after K rounds.

5. **The general framework solves a harder problem than we need.** The
   thesis handles arbitrary linear imprecise bandits with abstract
   bilinear maps. Our problem is specifically: learn an unknown K×K
   payoff matrix from (predictor_action, agent_action, reward)
   observations.

### 1.2 The Key Insight

Through discussion we identified that the core IUCB algorithm has a
simple interpretation:

- **UCB** maintains one confidence interval per arm (K intervals).
- **IUCB** maintains a confidence set over the full K×K payoff matrix.

The thesis implements this confidence set using abstract machinery (spaces
Z, W, Z_bar; bilinear map F; custom norms; slab constraints from cycles).
But for payoff matrix games, there is a much more direct approach:

- Maintain one confidence interval per cell P[b,a] (K² intervals).
- Use Hoeffding bounds, exactly as UCB does per arm.
- Compute the optimistic game value by solving a standard minimax game
  with the upper confidence bound matrix.
- Play the corresponding minimax strategy.

No cycles, no custom norms, no bilinear maps, no abstract spaces. Just
per-cell statistics, Hoeffding, and a game solver.

### 1.3 The Environment Model

The agent interacts with an environment that has the following structure:

- There are K agent actions (arms) and K predictor actions.
- There is a fixed, unknown payoff matrix P where P[b, a] is the reward
  when the predictor plays b and the agent plays a.
- Each round:
  1. The agent declares a mixed strategy x (probability distribution
     over actions).
  2. The predictor observes x and chooses action b. The predictor's
     strategy is unknown — it could be adversarial, cooperative,
     random, or anything else.
  3. The agent samples action a from x.
  4. The agent observes the full outcome: (b, a, reward = P[b,a]).
- Rewards may be stochastic: P[b,a] is the mean, and observed rewards
  have bounded noise. (In our current environments, rewards are
  deterministic given (b,a), but the algorithm handles noise.)

The agent's goal: minimize **minimax regret**, defined as:

    Regret(N) = N * V(P) - Σ_{t=1}^{N} reward_t

where V(P) = max_x min_b Σ_a x[a]*P[b,a] is the minimax game value
of the true payoff matrix.

This regret definition compares against the best fixed minimax strategy,
not against the best response to the actual predictor. This means:
- If the predictor is adversarial, the benchmark is tight.
- If the predictor is cooperative, the agent may do BETTER than the
  benchmark (negative regret), which is fine.

---

## 2. The Simplified Algorithm

### 2.1 Intuition

Think of it as "UCB for games":

1. **Estimate each cell of the payoff matrix** from observations, just
   like UCB estimates each arm's mean from pulls.

2. **Be optimistic**: construct a payoff matrix where each cell is set
   to its upper confidence bound. This is the most favorable game
   that's still plausible.

3. **Play the minimax strategy** for this optimistic game. This
   balances exploration (uncertain cells have wide confidence bounds,
   pulling the optimistic game toward strategies that test them) with
   exploitation (well-estimated cells guide toward the true optimum).

4. **Update cell estimates** from observed (b, a, reward) tuples.
   Repeat.

The optimism drives exploration: cells with few observations have wide
confidence intervals, making the optimistic payoff matrix assign high
values to strategies that would observe those cells. As all cells are
observed, the optimistic game converges to the true game, and the
minimax strategy converges to the true minimax strategy.

### 2.2 Pseudocode

```
class SimpleIUCBAgent:

    def reset():
        for each (b, a) pair:
            cell_sum[b, a] = 0       # sum of observed rewards
            cell_count[b, a] = 0     # number of observations
        total_rounds = 0

    def get_probabilities() -> x:
        P_upper = compute_upper_confidence_matrix()
        x = solve_minimax_strategy(P_upper)
        return x

    def update(b, a, reward):
        cell_sum[b, a] += reward
        cell_count[b, a] += 1
        total_rounds += 1

    def compute_cell_mean(b, a) -> float:
        if cell_count[b, a] == 0:
            return 0.0  # prior: no information
        return cell_sum[b, a] / cell_count[b, a]

    def compute_cell_bonus(b, a) -> float:
        if cell_count[b, a] == 0:
            return 1.0  # maximum uncertainty (rewards in [-1,1])
        return sqrt(2 * ln(total_rounds) / cell_count[b, a])

    def compute_upper_confidence_matrix() -> P_upper:
        for each (b, a):
            P_upper[b, a] = min(
                compute_cell_mean(b, a) + compute_cell_bonus(b, a),
                1.0  # clip to valid range
            )
        return P_upper

    def solve_minimax_strategy(P) -> x:
        # max_x min_b Σ_a x[a] * P[b, a]
        # For 2x2: closed-form (existing solve_2x2_game)
        # For K>2: linear program
        value, x = solve_game(P)
        return x
```

### 2.3 Key Design Decisions

**No cycles.** The strategy updates every round. Each observation
immediately improves one cell's estimate. There is no waiting for a
stopping condition.

**Per-cell Hoeffding bounds.** Each cell's confidence interval is
independent. The bonus term sqrt(2*ln(t)/n_{b,a}) comes directly from
the Hoeffding inequality, giving:

    P(|cell_mean - P[b,a]| > bonus) <= 2/t²

With a union bound over K² cells: all intervals hold simultaneously
with probability >= 1 - 2K²/t².

**Optimistic game value.** Using the upper confidence bound matrix is
optimistic because:
- The true P[b,a] <= P_upper[b,a] (with high probability)
- Therefore V(P) <= V(P_upper) (game value is monotone in payoffs
  when you take upper bounds in all cells)
- So the optimistic game value overestimates the true game value,
  which is the standard requirement for optimism-based algorithms

**Reward scaling.** Rewards must be in a known bounded range [r_min,
r_max] for Hoeffding to apply. This is a parameter (same as the full
IUCB). The scaling affects the bonus magnitude but not the algorithm
structure.

---

## 3. Detailed Method Specifications

### 3.1 Cell Statistics (`CellStats`)

Tracks per-cell observation statistics.

```python
class CellStats:
    """Statistics for one cell (b, a) of the payoff matrix."""

    def __init__(self):
        self.count = 0      # number of observations
        self.sum = 0.0      # sum of (scaled) rewards
        self.sum_sq = 0.0   # sum of squared rewards (for variance, optional)

    def update(self, reward_scaled: float):
        self.count += 1
        self.sum += reward_scaled
        self.sum_sq += reward_scaled ** 2

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def variance(self) -> float:
        """Sample variance (for potential Bernstein-style bounds)."""
        if self.count < 2:
            return 1.0  # maximum for [-1, 1]
        mean = self.mean()
        return self.sum_sq / self.count - mean ** 2
```

### 3.2 Confidence Matrix (`compute_confidence_matrix`)

Builds the upper confidence bound payoff matrix.

```python
def compute_confidence_matrix(
    cells: dict[(int, int), CellStats],
    num_actions: int,
    total_rounds: int,
    confidence_scale: float = 2.0,
) -> np.ndarray:
    """Compute the optimistic (upper confidence bound) payoff matrix.

    For each cell, UCB = mean + sqrt(confidence_scale * ln(t) / n).
    Unobserved cells get UCB = 1.0 (maximum possible).
    All values clipped to [-1, 1].

    Arguments:
        cells: per-cell statistics
        num_actions: K (number of actions for agent and predictor)
        total_rounds: total observations so far (t)
        confidence_scale: multiplier for the bonus (default 2.0 for Hoeffding)

    Returns:
        P_upper: shape (K, K), the optimistic payoff matrix
    """
    P_upper = np.ones((num_actions, num_actions))  # optimistic default

    if total_rounds < 1:
        return P_upper

    log_t = np.log(total_rounds)

    for b in range(num_actions):
        for a in range(num_actions):
            stats = cells.get((b, a))
            if stats is None or stats.count == 0:
                P_upper[b, a] = 1.0  # no info -> maximally optimistic
            else:
                bonus = np.sqrt(confidence_scale * log_t / stats.count)
                P_upper[b, a] = min(stats.mean() + bonus, 1.0)

    return P_upper
```

### 3.3 Game Solver (`solve_minimax_strategy`)

Computes the minimax strategy for a given payoff matrix.

```python
def solve_minimax_strategy(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve max_x min_b Σ_a x[a] * P[b, a].

    For 2x2: use existing closed-form solver (solve_2x2_game).
    For K>2: use linear programming.

    Arguments:
        P: shape (K, K), payoff matrix

    Returns:
        (value, x): game value and optimal mixed strategy
    """
    K = P.shape[0]
    if K == 2:
        return solve_2x2_game(P)
    else:
        return solve_game_lp(P)


def solve_game_lp(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve a K×K zero-sum game via linear programming.

    The LP formulation:
        max v
        s.t. Σ_a x[a] * P[b, a] >= v   for all b
             Σ_a x[a] = 1
             x[a] >= 0                   for all a

    Arguments:
        P: shape (K, K)

    Returns:
        (value, x): game value and optimal mixed strategy
    """
    # Use scipy.optimize.linprog or similar
    ...
```

### 3.4 The Agent (`SimpleIUCBAgent`)

The main agent class, following the existing BaseAgent interface.

```python
class SimpleIUCBAgent(BaseAgent):
    """Simplified IUCB agent using per-cell confidence intervals.

    Arguments:
        num_actions:      K (same for agent and predictor)
        reward_range:     (min, max) for scaling rewards to [-1, 1]
        confidence_scale: multiplier for Hoeffding bonus (default 2.0)
    """

    def __init__(self, *args,
                 reward_range: tuple[float, float] | None = None,
                 confidence_scale: float = 2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_range = reward_range
        self.confidence_scale = confidence_scale

    def reset(self):
        super().reset()
        self.cells = {}  # (b, a) -> CellStats
        self.total_rounds = 0
        for b in range(self.num_actions):
            for a in range(self.num_actions):
                self.cells[(b, a)] = CellStats()

    def get_probabilities(self) -> np.ndarray:
        P_upper = compute_confidence_matrix(
            self.cells, self.num_actions,
            self.total_rounds, self.confidence_scale
        )
        _, x = solve_minimax_strategy(P_upper)
        return x

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        reward_scaled = self._scale_reward(outcome.reward)
        b = outcome.env_action
        if b is not None:
            self.cells[(b, action)].update(reward_scaled)
        self.total_rounds += 1

    def _scale_reward(self, reward: float) -> float:
        if self.reward_range is None:
            return np.clip(reward, -1.0, 1.0)
        r_min, r_max = self.reward_range
        if r_max <= r_min:
            return 0.0
        return np.clip(2 * (reward - r_min) / (r_max - r_min) - 1, -1, 1)
```

### 3.5 Handling Standard Bandits (No Predictor)

For standard bandits, there is no predictor (env_action is None). The
payoff matrix degenerates: only the diagonal matters (b is meaningless).
In this case, the agent should fall back to standard UCB behavior.

```python
def get_probabilities(self) -> np.ndarray:
    if self._is_standard_bandit():
        return self._ucb_policy()
    P_upper = compute_confidence_matrix(...)
    _, x = solve_minimax_strategy(P_upper)
    return x

def _is_standard_bandit(self) -> bool:
    """True if no env_action has ever been observed."""
    return all(
        self.cells[(b, a)].count == 0
        for b in range(self.num_actions)
        for a in range(self.num_actions)
        if b != a  # only off-diagonal cells
    ) and self.total_rounds > 0

def _ucb_policy(self) -> np.ndarray:
    """Standard UCB1 policy for bandit case."""
    # Use diagonal cells as arm estimates
    ucb_values = np.zeros(self.num_actions)
    for a in range(self.num_actions):
        stats = self.cells[(a, a)]  # diagonal: b == a
        if stats.count == 0:
            ucb_values[a] = float('inf')
        else:
            bonus = np.sqrt(2 * np.log(self.total_rounds) / stats.count)
            ucb_values[a] = stats.mean() + bonus
    best = np.argmax(ucb_values)
    x = np.zeros(self.num_actions)
    x[best] = 1.0
    return x
```

---

## 4. Implementation Tasks

### Phase 1: Core Agent (New File)

**File**: `ibrl/agents/simple_iucb.py`

1. Create `CellStats` class (as specified in 3.1)
2. Create `compute_confidence_matrix` function (as in 3.2)
3. Create `SimpleIUCBAgent` class (as in 3.4)
   - `__init__`, `reset`, `get_probabilities`, `update`, `_scale_reward`
   - Reuse existing `solve_2x2_game` from `ibrl/utils/game_solving.py`
4. Handle standard bandit fallback (as in 3.5)

**No changes needed** to algebra.py, confidence_set.py, or any of the
existing IUCB machinery. The simplified agent is a completely independent
implementation.

### Phase 2: Game Solver for K > 2 (Optional)

**File**: `ibrl/utils/game_solving.py`

5. Add `solve_game_lp` for K×K games using scipy.optimize.linprog
   - Only needed if we want to support K > 2
   - 2×2 already has a closed-form solver

### Phase 3: Registration and Construction

**File**: `ibrl/agents/__init__.py`

6. Export `SimpleIUCBAgent`

**File**: `ibrl/utils/construction.py`

7. Register as `"simple-iucb"` in the agent_types dict

### Phase 4: Tests

**File**: `tests/test_simple_iucb_agent.py`

8.  Unit test: construction via string `"simple-iucb"`
9.  Unit test: reset initializes K² cells with count 0
10. Unit test: update increments correct cell
11. Unit test: unobserved cells have UCB = 1.0
12. Unit test: observed cells have correct mean and bonus
13. Unit test: minimax strategy is computed from UCB matrix
14. Integration test: Damascus — converges to p≈0.5, reward > 1.0
    (should work in ~100 steps, unlike full IUCB which needs 500+)
15. Integration test: Newcomb — achieves reward > 5.0
16. Integration test: Bandit — sublinear regret
17. Comparison test: SimpleIUCB vs full IUCB on Damascus at N=100
    (SimpleIUCB should be substantially better)

### Phase 5: Notebook Update

**File**: `experiments/alaro/example.ipynb`

18. Add `"simple-iucb"` to the agents list
19. Compare simple-iucb vs iucb vs ucb across all environments
20. Add a cell showing convergence speed comparison

### Phase 6: Cleanup

21. Update `20260320_ibplan.md` Section 15.4 to note the simplified
    alternative exists
22. Verify all existing tests still pass (63 tests for full IUCB)

---

## 5. What Changes vs. the Full IUCB

| Aspect | Full IUCB | Simplified IUCB |
|--------|-----------|-----------------|
| Hypothesis space | 10D Z (embedded payoff matrix) | 4D payoff matrix directly |
| Confidence set | Slab constraints from cycles | Per-cell Hoeffding intervals (box) |
| Update frequency | Once per cycle (~100+ rounds) | Every round |
| Policy computation | Optimistic θ → minimax x | UCB matrix → minimax x |
| Norms | Custom Z_bar norm (operator norm) | None needed |
| Stopping condition | sqrt(τ)*max_dist >= threshold | None (no cycles) |
| Lines of code | ~500 (algebra + confidence_set + agent) | ~100 (agent only) |
| Dependencies | scipy.optimize (for norms, L-BFGS-B) | solve_2x2_game only |
| Min useful N | ~500 (2×2 games) | ~10-20 (just need a few obs per cell) |

---

## 6. Regret Bound Analysis

### 6.1 Why a Formal Regret Bound Should Be Achievable

The simplified algorithm is a direct application of the "optimism in the
face of uncertainty" (OFU) principle, which is the standard framework for
proving regret bounds in bandits and games. The key ingredients:

**Ingredient 1: Valid confidence intervals.**
Each cell P[b,a] is estimated from i.i.d. observations (conditional on
(b,a) occurring). Hoeffding's inequality gives:

    P(|P̂[b,a] - P[b,a]| > sqrt(2*ln(t)/n_{b,a})) <= 2/t²

Union bound over K² cells: all intervals hold simultaneously with
probability >= 1 - 2K²/t². This gives a valid confidence box that
contains the true P with high probability.

These intervals are valid regardless of the predictor's strategy. The
predictor controls WHICH cells get observed (by choosing b), but cannot
affect the reward distribution within a cell. So even an adversarial
predictor cannot invalidate the confidence intervals.

**Ingredient 2: Optimism.**
The UCB payoff matrix P_upper satisfies P_upper[b,a] >= P[b,a] for all
(b,a) with high probability. Therefore:

    V(P_upper) >= V(P)

where V(·) is the minimax game value. This is because the game value
is monotone non-decreasing when all payoffs increase (the agent can
only benefit from higher payoffs).

The agent plays the minimax strategy for P_upper, so its expected
reward (against a worst-case predictor) is at least V(P_upper) minus
the gap caused by P_upper != P.

**Ingredient 3: Regret decomposition.**
Per-round regret is bounded by the gap between V(P_upper) and the
actual reward. This gap has two sources:

a) **Estimation error**: P_upper[b,a] - P[b,a] = O(sqrt(ln(t)/n_{b,a}))
   for observed cells. This shrinks as cells are observed.

b) **Exploration cost**: cells with few observations have wide confidence
   intervals. The optimistic game value may overestimate because of
   these cells.

The standard OFU argument then shows: the cumulative gap is bounded
by O(K² * sqrt(N * ln(N))), giving O~(sqrt(N)) regret in the
worst case, with constants depending on K (not on D_Z=10 or R=12).

**Ingredient 4: Exploration is guaranteed.**
The optimism mechanism ensures exploration. If a cell (b,a) has a wide
confidence interval, P_upper[b,a] is high. This makes the optimistic
game value favor strategies where that cell could be realized. The
agent will play strategies that induce the predictor to reveal
under-observed cells.

For the adversarial predictor case: the predictor controls b, so it
controls which row is observed. But the predictor is playing a zero-sum
game — it picks b to minimize the agent's reward. If the predictor
avoids a row, that row isn't the worst case, so it doesn't affect the
minimax value. Unobserved rows have wide confidence intervals, making
them look "good" in the optimistic matrix, but the adversary would
only avoid them if they're truly not harmful — in which case the agent
doesn't need to learn them.

The circular argument ("unobserved cells don't matter because the
adversary would use them if they did") needs careful formalization.
For 2×2, the combinatorics are small enough that this can be made
rigorous. For arbitrary K, it requires more work but the structure
is the same.

### 6.2 Expected Regret Bound

For a K×K payoff matrix game, the simplified algorithm should achieve:

**General case**: O(K * sqrt(N * ln(N)))

**Positive gap case** (when the optimal minimax strategy is unique and
has gap Δ > 0): O(K² * ln(N) / Δ)

These bounds have constants that depend polynomially on K with small
exponents, in contrast to the thesis's K^{17/3} for the general case.

### 6.3 What We Lose vs. the Full IUCB

**1. Generality.** The thesis's algorithm works for arbitrary linear
imprecise bandits — not just payoff matrices. If the hypothesis space
has a more complex structure (e.g., the credal set can't be decomposed
into "pick a row"), the per-cell approach doesn't apply. The thesis's
bilinear map F and custom norms handle these general structures.

In practice, for our Newcomb-like games, this generality is unnecessary.
The environment IS a payoff matrix game.

**2. Handling genuinely imprecise outcomes.** In the general framework,
even knowing the hypothesis θ doesn't pin down the outcome distribution
(the credal set K_θ(x) has multiple elements). The per-cell approach
assumes that P[b,a] is a fixed (possibly stochastic) value — there's
no additional adversarial freedom within a cell.

For our environments, this is true: given (b, a), the reward is
deterministic (or has simple i.i.d. noise). There's no hidden adversary
within a cell.

**3. The formal regret proof doesn't exist yet.** The thesis provides a
complete, published proof for the full IUCB. The simplified algorithm's
regret bound is conjectured based on standard techniques but has not
been formally proven. The main technical challenge is the adversarial
exploration argument (Section 6.1, Ingredient 4).

For the cooperative predictor case (predictor samples b ~ x), the proof
is straightforward — standard OFU + Hoeffding. For the fully adversarial
case, the exploration argument needs more work, but uses the same
techniques as existing game-theoretic bandit literature.

**4. Finite-time constants.** The thesis's bound has enormous but
explicitly computed constants. The simplified bound's constants
haven't been computed. They should be much smaller (polynomial in K
with small exponents vs K^{17/3}), but this needs verification.

### 6.4 What We Gain

**1. Practical regret bound.** The bound should hold at realistic N
(hundreds of steps), not N >> 20M.

**2. No heuristic parameters.** The confidence_scale=2.0 comes directly
from Hoeffding (not a heuristic). There is no eta to tune, no
time_horizon to set, no cycle length to worry about.

**3. Immediate adaptation.** The agent updates after every observation,
not once per cycle. This means it can adapt in O(K²) rounds (once
each cell has been observed), not O(cycle_length) rounds.

**4. Simplicity.** ~100 lines of code vs ~500. One file vs three
(algebra.py + confidence_set.py + iucb.py). No custom norms, no
bilinear maps, no kernel computations, no operator norms.

**5. Transparency.** The algorithm's behavior is easy to understand
and debug. You can inspect the confidence matrix at any time and see
exactly why the agent is choosing its strategy.

### 6.5 Nonrealizability

The simplified approach handles nonrealizability well within the payoff
matrix class. The hypothesis space [-1,1]^{K²} is ALL possible payoff
matrices (after scaling). The true P is guaranteed to be in this space
as long as rewards are properly scaled. There is no prior to specify,
no hypothesis class to enumerate, no risk of the true environment being
outside the model.

This contrasts with frameworks like AIXI, which require an enumerable
hypothesis class and suffer catastrophically if the true environment
isn't included.

Where nonrealizability WOULD bite: if the environment isn't actually a
stationary payoff matrix game (e.g., non-stationary rewards, hidden
states, predictor with memory). Neither the simplified nor the full
IUCB handles these cases.

---

## 7. Open Questions

1. **Formal proof for adversarial predictor.** The exploration argument
   (unobserved cells don't affect the minimax value) needs a rigorous
   proof. This is the main theoretical gap.

2. **Empirical comparison at larger K.** The simplified approach should
   scale gracefully to K=5-10. The full IUCB's constants grow as
   K^{17/3}, making it impractical beyond K=2.

3. **Bernstein-style bounds.** The Hoeffding bonus sqrt(2*ln(t)/n) could
   be tightened using empirical variance (Bernstein's inequality). This
   would give tighter confidence intervals for cells with low variance,
   potentially improving performance. The CellStats class already tracks
   sum_sq for this purpose.

4. **Comparison with EXP3.** EXP3 handles adversarial rewards with
   O(sqrt(K*N*ln(K))) regret but doesn't exploit payoff matrix structure.
   SimpleIUCB exploits the structure (K² parameters, not arbitrary
   reward sequences). On Newcomb-like games, SimpleIUCB should dominate.

5. **Lower bound.** What is the minimax-optimal regret for learning
   K×K payoff matrix games? Is O(K*sqrt(N)) tight, or can it be improved
   to O(sqrt(K*N))? This would tell us whether the simplified algorithm
   is optimal or if further improvements are possible.

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

### 2.1 Provenance

This algorithm is **Algorithm 1 ("UCB for matrix games") from
O'Donoghue, Lattimore, and Osband (2021), "Matrix games with bandit
feedback," UAI 2021**
([arXiv](https://arxiv.org/abs/2006.05145),
[PMLR](https://proceedings.mlr.press/v161/o-donoghue21a.html)),
with a minor variant of the confidence bonus term.

A closely related algorithm, **Maximin-UCB (Algorithm 2) from Ito,
Luo, Maiti, Tsuchiya, and Wu (2026), "Adversarial Learning in Games
with Bandit Feedback"**
([arXiv](https://arxiv.org/abs/2602.06348)), differs from
O'Donoghue et al. only in playing a **pure** maximin strategy instead
of a **mixed** one, and uses a different regret benchmark (pure-
strategy maximin regret instead of Nash regret). We follow O'Donoghue
et al. in playing mixed strategies, because our games (e.g. Damascus)
require mixed equilibria.

### 2.2 Relationship to the Published Algorithms

The three algorithms side by side:

**O'Donoghue et al. (2021), Algorithm 1:**
```
for round t = 1, 2, ..., T:
    Ã_ij = Ā_ij + sqrt(2 * log(2 * T² * m * k) / (1 ∨ n_ij))
    x = argmax_{x ∈ Δk} min_{y ∈ Δm} y^T Ã x      [mixed strategy]
    play, observe opponent action and noisy reward
    update Ā_ij and n_ij for the observed cell
```

**Ito et al. (2026), Algorithm 2 (Maximin-UCB):**
```
for round t = 1, 2, ...:
    U(x,y) = R(x,y)/N(x,y) + sqrt((4*log(T) + 2*log(1+N(x,y))) / N(x,y))
             or 1 if N(x,y) = 0
    x(t) = argmax_{x ∈ X} min_{y ∈ Y} U(x,y)       [pure strategy]
    play x(t), observe y(t) and reward r_t
    update R(x(t),y(t)) and N(x(t),y(t))
```

**Our plan:**
```
for each round:
    P_upper[b,a] = mean[b,a] + sqrt(2 * ln(t) / count[b,a])
                   clipped to 1.0; set to 1.0 if count = 0
    x = argmax_x min_b Σ_a x[a] * P_upper[b,a]      [mixed strategy]
    play a ~ x, observe (b, a, reward)
    update cell_sum[b,a], cell_count[b,a]
```

**Specific differences from O'Donoghue et al.:**

1. **Bonus term**: We use `sqrt(2 * ln(t) / n)` (standard UCB1,
   anytime). They use `sqrt(2 * log(2T²mk) / n)` (fixed-horizon,
   requires knowing T in advance). Both are valid Hoeffding bounds;
   ours avoids requiring the time horizon as input. The regret bound
   changes by at most a log factor.

2. **Reward range**: They use [0, 1]. We scale to [-1, 1] via
   `_scale_reward`. This is cosmetic — the Hoeffding bound applies
   to any known bounded range.

3. **Unobserved cells**: They set Ā_ij = 0 with a bonus ≥ 1
   (so UCB ≥ 1). We set mean = 0.0, bonus = 1.0, so UCB = 1.0.
   Same effect: maximally optimistic for unobserved cells.

**Everything else is identical**: per-cell Hoeffding intervals,
construct the full UCB payoff matrix, solve the minimax game on the
UCB matrix to get a mixed strategy, update only the observed cell
each round, per-round updates (no cycles/batching).

**The Õ(√(mkT)) worst-case Nash regret bound from O'Donoghue et al.
Theorem 1 applies to our algorithm** with at most a log-factor
difference due to the bonus term variant.

### 2.3 Intuition

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

### 2.4 Pseudocode

```
class MatrixUCBAgent:

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
        # LP via scipy.optimize.linprog (works for any K)
        value, x = solve_minimax_strategy(P)
        return x
```

### 2.5 Key Design Decisions

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

Computes the minimax strategy for a given payoff matrix via linear
programming. Works for any K×K game (including 2×2).

```python
from scipy.optimize import linprog

def solve_minimax_strategy(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve max_x min_b Σ_a x[a] * P[b, a] via LP.

    The LP (maximizing v is equivalent to minimizing -v):
        min  -v
        s.t. Σ_a x[a] * P[b, a] >= v   for all b
             Σ_a x[a] = 1
             x[a] >= 0                   for all a

    Decision variables: [x[0], ..., x[K-1], v].

    Arguments:
        P: shape (K, K), payoff matrix (rows = predictor, cols = agent)

    Returns:
        (value, x): game value and optimal mixed strategy
    """
    K = P.shape[0]

    # Objective: minimize -v. Variables are [x_0, ..., x_{K-1}, v].
    c = np.zeros(K + 1)
    c[K] = -1.0  # minimize -v

    # Inequality constraints: v - Σ_a x[a] * P[b,a] <= 0 for each b.
    # (Rewritten from P^T x >= v*1 to v*1 - P^T x <= 0.)
    A_ub = np.zeros((K, K + 1))
    A_ub[:, :K] = -P        # -P[b, a] * x[a]
    A_ub[:, K] = 1.0        # +v
    b_ub = np.zeros(K)

    # Equality constraint: Σ_a x[a] = 1.
    A_eq = np.zeros((1, K + 1))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x[a] >= 0, v is unbounded.
    bounds = [(0, None)] * K + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    x = result.x[:K]
    value = result.x[K]
    return value, x
```

### 3.4 The Agent (`MatrixUCBAgent`)

The main agent class, following the existing BaseAgent interface.

```python
class MatrixUCBAgent(BaseAgent):
    """UCB for matrix games with unknown payoffs.

    Implements Algorithm 1 from O'Donoghue, Lattimore, and Osband
    (2021), "Matrix games with bandit feedback," UAI 2021.
    https://arxiv.org/abs/2006.05145

    Each round, the agent constructs an upper confidence bound (UCB)
    payoff matrix from per-cell Hoeffding intervals, then plays the
    mixed minimax strategy of this optimistic matrix. Observes the
    opponent's action and reward, updates the corresponding cell.

    This is also a simplification of Kosoy's IUCB algorithm (2024,
    "Imprecise Multi-Armed Bandits," https://arxiv.org/abs/2405.05673)
    to the payoff matrix special case. Kosoy's general framework
    handles arbitrary linear imprecise bandits using cycles, abstract
    norms, and slab-based confidence sets in a 10D hypothesis space.
    For payoff matrix games, all of that machinery reduces to per-cell
    confidence intervals and a minimax solve — this algorithm.

    Differences from O'Donoghue et al. Algorithm 1:
    - Bonus term: we use sqrt(2 * ln(t) / n) (anytime UCB1) instead
      of sqrt(2 * log(2T²mk) / n) (fixed-horizon). Avoids requiring
      the time horizon T as input; costs at most a log factor in the
      regret bound.
    - Reward range: we scale to [-1, 1]; they use [0, 1]. Cosmetic.

    Regret bound (O'Donoghue et al. Theorem 1):
      WorstCaseRegret <= O~(sqrt(m * k * T))
    where m, k are action counts and T is the number of rounds.

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
        # Default env_action to 0 for standard bandits (no predictor).
        # This places all observations in row 0 of the K×K matrix.
        b = outcome.env_action if outcome.env_action is not None else 0
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

### 3.5 Standard Bandits: Graceful Degeneration

For standard bandits (env_action is always None), the algorithm
requires no special case. With env_action defaulting to 0, all
observations go into row 0 of the K×K UCB matrix. The minimax
computation then naturally reduces to UCB1:

- Row 0 has real UCB values from observations.
- Rows 1..K-1 remain at 1.0 (never observed, maximally optimistic).
- `min_b` picks row 0 (the only row with values < 1.0 once cells
  are observed).
- `max_x min_b` maximizes row 0's expected value = pick the action
  with the highest UCB in row 0 = standard UCB1.

No `_is_standard_bandit()` check or `_ucb_policy()` fallback needed.

---

## 4. Implementation Tasks

### Phase 1: Core Agent (New File)

**File**: `ibrl/agents/matrix_ucb.py`

1. Create `CellStats` class (as specified in 3.1)
2. Create `compute_confidence_matrix` function (as in 3.2)
3. Create `solve_minimax_strategy` function (as in 3.3)
   - Uses `scipy.optimize.linprog` (HiGHS). Works for any K.
4. Create `MatrixUCBAgent` class (as in 3.4)
   - `__init__`, `reset`, `get_probabilities`, `update`, `_scale_reward`
   - Standard bandits handled by defaulting env_action to 0 (see 3.5)

**No changes needed** to algebra.py, confidence_set.py, or any of the
existing IUCB machinery. The simplified agent is a completely independent
implementation.

### Phase 2: Registration and Construction

**File**: `ibrl/agents/__init__.py`

5. Export `MatrixUCBAgent`

**File**: `ibrl/utils/construction.py`

6. Register as `"matrix-ucb"` in the agent_types dict

### Phase 3: Tests

**File**: `tests/test_simple_iucb_agent.py`

7.  Unit test: construction via string `"matrix-ucb"`
8.  Unit test: reset initializes K² cells with count 0
9.  Unit test: update increments correct cell
10. Unit test: unobserved cells have UCB = 1.0
11. Unit test: observed cells have correct mean and bonus
12. Unit test: minimax strategy is computed from UCB matrix
13. Integration test: Damascus — converges to p≈0.5, reward > 1.0
    (should work in ~100 steps, unlike full IUCB which needs 500+)
14. Integration test: Newcomb — achieves reward > 5.0
15. Integration test: Bandit — sublinear regret
16. Comparison test: SimpleIUCB vs full IUCB on Damascus at N=100
    (SimpleIUCB should be substantially better)

### Phase 4: Notebook Update

**File**: `experiments/alaro/example.ipynb`

17. Add `"matrix-ucb"` to the agents list
18. Compare matrix-ucb vs iucb vs ucb across all environments
19. Add a cell showing convergence speed comparison

### Phase 5: Cleanup

20. Update `20260320_ibplan.md` Section 15.4 to note the simplified
    alternative exists
21. Verify all existing tests still pass (63 tests for full IUCB)

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
| Lines of code | ~500 (algebra + confidence_set + agent) | ~80 (agent + LP solver) |
| Dependencies | scipy.optimize (for norms, L-BFGS-B) | scipy.optimize.linprog only |
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

---

## 8. Prior Work and Literature Review

**Date**: 2026-03-22

Our "UCB for games" approach — maintaining per-cell confidence intervals
on the payoff matrix and playing the minimax strategy of the optimistic
(upper confidence bound) matrix — is a known technique. It has been
studied under several names and is a standard specialization of
model-based optimistic RL (UCRL) to the single-state game setting.
Below is a summary of the most relevant prior work, organized by
relevance to our setting.

### 8.1 Directly Relevant: UCB for Matrix Games

**O'Donoghue, Lattimore, Osband (2021). "Matrix games with bandit
feedback."** *UAI 2021 (Proceedings of the 37th Conference on
Uncertainty in Artificial Intelligence).*
[arXiv](https://arxiv.org/abs/2006.05145) |
[PMLR](https://proceedings.mlr.press/v161/o-donoghue21a.html)

This is the single most relevant paper. It studies exactly our
problem: a zero-sum matrix game with an unknown payoff matrix, where
players observe each other's actions and a noisy payoff. The authors
propose and analyze UCB variants that maintain confidence intervals on
the payoff matrix entries and use them to select strategies. Key results:

- UCB achieves Õ(√(K²T)) *Nash regret* (regret against the minimax
  game value), where K is the number of actions per player.
- A variant called "K-learning" achieves similar bounds with a
  different exploration mechanism.
- Thompson Sampling **fails catastrophically** in this setting — an
  important negative result.
- EXP3 (adversarial bandits) achieves low regret but is empirically
  much worse than UCB because it does not exploit the matrix game
  structure.
- The UCB and K-learning agents "learn to play Nash strategies and are
  not exploitable by any opponent," even when the opponent adversarially
  plays best-response to the learner's mixed strategy.

**Relationship to our approach**: Our simplified IUCB is essentially
this paper's UCB algorithm. The main difference is framing: they study
it as a standalone game-learning algorithm, while we arrived at it as
a simplification of Kosoy's general imprecise bandit framework.

**Ito, Luo, Tsuchiya, Wu (2026). "Adversarial Learning in Games with
Bandit Feedback: Logarithmic Pure-Strategy Maximin Regret."**
[arXiv](https://arxiv.org/abs/2602.06348)

This very recent paper proposes **Maximin-UCB**, which is described as
playing "the pure maximin strategy of the upper confidence bound of the
unknown game matrix" — essentially exactly our algorithm. Key results:

- In the *informed* setting (observing opponent's action, like our
  setting), Maximin-UCB achieves O(c' log T) instance-dependent regret,
  where c' is a game-dependent constant.
- In the *uninformed* setting (bandit feedback only), the Tsallis-INF
  algorithm achieves O(c log T) instance-dependent regret.
- Both results are for "pure-strategy maximin regret" (PSMR), which
  coincides with Nash regret when the game has a pure-strategy NE.
- Results generalize to bilinear games over arbitrary action sets
  (Maximin-LinUCB).

**Relationship to our approach**: Maximin-UCB IS our approach, with a
formal analysis. The O(log T) instance-dependent bound is much
stronger than the Õ(√T) worst-case bound, applicable when there is
a gap (the optimal pure strategy is strictly better than alternatives).

**Lin (2025). "Randomised Optimism via Competitive Co-Evolution for
Matrix Games with Bandit Feedback."**
[arXiv](https://arxiv.org/abs/2505.13562)

Proposes a randomized variant of optimistic matrix game learning using
evolutionary algorithms. Confirms that UCB (deterministic optimism)
achieves Õ(√(K²T)) Nash regret, and shows that randomized optimism
matches this rate. Mainly relevant for confirming the regret bound of
the UCB approach.

### 8.2 Instance-Dependent and Logarithmic Regret

**Maiti, Jamieson, Ratliff (2023/2025). "On the Limitations and
Possibilities of Nash Regret Minimization in Zero-Sum Matrix Games
under Noisy Feedback."**
[arXiv](https://arxiv.org/abs/2306.13233) |
[Springer](https://link.springer.com/chapter/10.1007/978-3-032-03639-1_1)

Key negative and positive results:

- **Negative**: Standard algorithms (Hedge, FTRL, OMD) and even
  playing the Nash equilibrium of the empirical payoff matrix all
  incur Ω(√T) Nash regret, even with full-information feedback.
- **Positive**: The first algorithm for general n×m matrix games
  achieving instance-dependent polylog(T) Nash regret in the
  full-information setting.
- For 2×2 games with bandit feedback, they also achieve polylog(T)
  Nash regret — in a regime where existing algorithms provably suffer
  Ω(√T).

This suggests that our UCB-based approach, which uses the full payoff
matrix structure (not just the empirical NE), is on the right track.
It avoids the Ω(√T) lower bound that applies to empirical-NE methods
by using optimistic confidence bounds.

**Ito, Luo, Tsuchiya, Wu (2025). "Instance-Dependent Regret Bounds
for Learning Two-Player Zero-Sum Games with Bandit Feedback."**
[arXiv](https://arxiv.org/abs/2502.17625) |
[PMLR](https://proceedings.mlr.press/v291/ito25a.html)

Shows that Tsallis-INF achieves O(c₁ log T + √(c₂T)) regret in
bandit feedback, where c₁ depends inversely on gap measures and c₂
can be much smaller than K when NE has small support. When a
pure-strategy NE exists, c₂ = 0, giving optimal O(log T) regret.

### 8.3 The Optimism-then-NoRegret Framework

**Li, Liu, Pu, Liang, Luo (2024). "Optimistic Thompson Sampling for
No-Regret Learning in Unknown Games."**
[arXiv](https://arxiv.org/abs/2402.09456)

Introduces the Optimism-then-NoRegret (OTN) framework for learning in
unknown games with bandit feedback. The framework encompasses UCB-based
algorithms as special cases. Key idea: first use optimism (UCB or
Thompson Sampling) to explore the unknown payoff structure, then
transition to no-regret play. Achieves regret bounds that depend
logarithmically on the total action space under specific reward
structures, alleviating the curse of multi-player games.

### 8.4 Model-Based RL for Markov Games (Superset of Our Setting)

Our payoff matrix game is the single-state (S=1), single-step (H=1)
special case of a two-player zero-sum Markov game. Several recent
papers study this more general setting using model-based optimistic
approaches that are direct generalizations of our algorithm.

**Bai and Jin (2020). "Provable Self-Play Algorithms for Competitive
Reinforcement Learning."** *ICML 2020.*
[arXiv](https://arxiv.org/abs/2002.04017) |
[PMLR](http://proceedings.mlr.press/v119/bai20a/bai20a.pdf)

Proposes VI-ULCB (Value Iteration with Upper/Lower Confidence Bound)
for two-player zero-sum Markov games. Achieves Õ(√T) regret. For the
matrix game special case (S=1, H=1), the regret bound reduces to
Õ(√(ABT)) where A, B are action counts.

**Liu, Yu, Bai, Jin (2021). "A Sharp Analysis of Model-based
Reinforcement Learning with Self-Play."** *ICML 2021.*
[arXiv](https://arxiv.org/abs/2010.01604) |
[PMLR](https://proceedings.mlr.press/v139/liu21z.html)

Proposes Optimistic Nash Value Iteration (Nash-VI). Achieves
Õ(H³SAB/ε²) sample complexity, matching the information-theoretic
lower bound Ω(H³S(A+B)/ε²) up to a min{A,B} factor. For matrix
games (S=1, H=1), this gives Õ(AB/ε²) — confirming that the
confidence-set + optimistic-minimax approach is near-optimal.

**Zhang, Ji, Du, Wang (2020). "Model-Based Multi-Agent RL in Zero-Sum
Markov Games with Near-Optimal Sample Complexity."** *NeurIPS 2020.*
[arXiv](https://arxiv.org/abs/2007.07461) |
[JMLR (2023)](https://www.jmlr.org/papers/volume24/20-1131/20-1131.pdf)

Achieves minimax-optimal (up to log factors) sample complexity for
model-based MARL in zero-sum Markov games using confidence sets and
optimistic planning.

### 8.5 Foundational Optimistic RL (UCRL2)

**Jaksch, Ortner, Auer (2010). "Near-optimal Regret Bounds for
Reinforcement Learning."** *JMLR 11:1563–1600.*
[JMLR](https://jmlr.org/papers/v11/jaksch10a.html) |
[PDF](https://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf)

The foundational paper on optimistic model-based RL. UCRL2 maintains
confidence sets over transition probabilities and rewards, then solves
the optimistic MDP. Achieves Õ(DS√(AT)) regret for MDPs with S states,
A actions, and diameter D. For the single-state case (S=1, D=1), this
reduces to Õ(√(AT)) — exactly the structure of our algorithm. Our
approach is a direct specialization of the UCRL2 principle to
matrix games.

### 8.6 Classical No-Regret Learning in Games

These are the baselines our approach should be compared against.

**Auer, Cesa-Bianchi, Fischer (2002). "Finite-time Analysis of the
Multiarmed Bandit Problem."** *Machine Learning 47:235–256.*
[Springer](https://link.springer.com/article/10.1023/A:1013689704352)

The original UCB1 paper. Our per-cell confidence intervals use the
same Hoeffding-based bonus term. UCB1 applies to independent arms;
our extension applies UCB thinking to each cell of a game matrix.

**Auer, Cesa-Bianchi, Freund, Schapire (2002). "The Nonstochastic
Multiarmed Bandit Problem."** *SICOMP 32(1):48–77.*
[DOI](https://doi.org/10.1137/S0097539701398375)

Introduces EXP3 for adversarial bandits. Achieves O(√(KT log K))
regret. EXP3 does not exploit matrix game structure and performs
empirically worse than UCB in matrix games (confirmed by O'Donoghue
et al. 2021). However, EXP3 handles non-stationary opponents, which
our approach does not.

**Cesa-Bianchi and Lugosi (2006). "Prediction, Learning, and Games."**
*Cambridge University Press.*
[CUP](https://doi.org/10.1017/CBO9780511546921)

The definitive reference on online learning and games. Chapter 7
establishes that if both players use Hannan-consistent (no-external-
regret) strategies, their average play converges to the minimax value.
The book focuses on adversarial methods (Hedge/MW) rather than
optimistic approaches like UCB.

**Lattimore and Szepesvári (2020). "Bandit Algorithms."** *Cambridge
University Press.*
[Free draft](https://banditalgs.com/) |
[CUP](https://doi.org/10.1017/9781108571401)

Comprehensive reference covering UCB, EXP3, linear bandits, and
adversarial bandits. Discusses game-theoretic interpretations. Notes
the fundamental incompatibility: UCB fails against adversarial
opponents (linear regret), while EXP3 pays √K overhead in stochastic
settings. "Best-of-both-worlds" algorithms (see Zimmert & Seldin
below) bridge this gap.

### 8.7 Best-of-Both-Worlds and Adaptive Algorithms

**Zimmert and Seldin (2019/2021). "Tsallis-INF: An Optimal Algorithm
for Stochastic and Adversarial Bandits."** *AISTATS 2019; JMLR 2021.*
[arXiv](https://arxiv.org/abs/1807.07623) |
[JMLR](https://jmlr.org/papers/v22/19-753.html)

Achieves optimal O(√(KT)) adversarial regret AND O(Σ log(T)/Δᵢ)
stochastic regret simultaneously. Relevant because it shows that
algorithms can adapt to the difficulty of the opponent without
knowing in advance whether the setting is stochastic or adversarial.
Used as a subroutine in recent game-learning papers (Ito et al. 2025,
2026).

**Syrgkanis, Agarwal, Luo, Schapire (2015). "Fast Convergence of
Regularized Learning in Games."** *NeurIPS 2015.*
[arXiv](https://arxiv.org/abs/1507.00407)

Shows that optimistic FTRL achieves O(T^{-3/4}) individual regret and
O(T^{-1}) convergence to equilibrium when used by both players —
quadratic improvement over standard O(T^{-1/2}). Relevant as it
demonstrates the power of optimism in game learning.

**Daskalakis, Deckelbaum, Kim (2011/2015). "Near-Optimal No-Regret
Algorithms for Zero-Sum Games."** *SODA 2011; Games and Economic
Behavior 2015.*
[PDF](https://people.csail.mit.edu/costis/optimalNR.pdf) |
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0899825614000049)

When used by both players, achieves O(ln T / T) convergence to the
game value — near-optimal. This is for the known-game setting, but
establishes the benchmark for convergence rates.

### 8.8 Unknown Games with Structured Payoffs

**Sessa, Bogunovic, Kamgarpour, Krause (2019). "No-Regret Learning in
Unknown Games with Correlated Payoffs."** *NeurIPS 2019.*
[arXiv](https://arxiv.org/abs/1909.08540) |
[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/685217557383cd194b4f10ae4b39eebf-Abstract.html) |
[Code](https://github.com/sessap/noregretgames)

Proposes GP-MW: uses Gaussian processes to model correlations between
payoff entries, combined with multiplicative weights. Achieves
kernel-dependent regret bounds. Relevant if payoff entries are
correlated (e.g., similar actions have similar payoffs). For our
small (2×2) games, the GP overhead is unnecessary, but the approach
could be relevant for larger games.

### 8.9 Blackwell Approachability

**Abernethy, Bartlett, Hazan (2011). "Blackwell Approachability and
No-Regret Learning are Equivalent."** *COLT 2011.*
[arXiv](https://arxiv.org/abs/1011.1936) |
[PMLR](https://proceedings.mlr.press/v19/abernethy11b.html)

Proves that Blackwell approachability and online linear optimization
are equivalent. This provides a game-theoretic foundation for
understanding why optimism-based approaches work in games: the
"optimism" principle maps to choosing approach strategies optimistically
in Blackwell's framework.

### 8.10 Relationship to Kosoy (2024)

**Kosoy (2024/2025). "Imprecise Multi-Armed Bandits: Representing
Irreducible Uncertainty as a Zero-Sum Game."**
*JMLR 26(184):1–75, 2025.*
[arXiv](https://arxiv.org/abs/2405.05673) |
[JMLR](http://jmlr.org/papers/v26/24-2001.html)

Our payoff matrix game is a special case of Kosoy's imprecise bandit
framework. The literature on matrix game learning (§8.1–8.4 above)
subsumes this special case with tighter bounds:

- Kosoy's general framework has regret constants scaling as K^{17/3}
  for the worst case. The matrix game literature achieves Õ(√(K²T))
  with much smaller constants.
- Kosoy's algorithm (IUCB) requires cycles, stopping conditions, and
  abstract norm computations. The matrix game algorithms (UCB,
  Maximin-UCB) are direct and per-round.
- The key theoretical contribution of Kosoy's work — handling genuinely
  imprecise outcomes where even knowing the hypothesis doesn't pin down
  the outcome distribution — is unnecessary for our payoff matrix
  setting.

The simplified IUCB we propose is essentially the standard algorithm
for this well-studied special case, with a direct lineage from UCRL2
through the Markov game literature.

---

## 9. Updated Regret Bound Analysis

Based on the literature review, we can sharpen our regret analysis:

### 9.1 Known Bounds for Our Setting

| Setting | Bound | Reference |
|---------|-------|-----------|
| Worst-case Nash regret (bandit) | Õ(√(K²T)) | O'Donoghue et al. 2021 |
| Worst-case Nash regret (informed) | Õ(√(K²T)) | O'Donoghue et al. 2021 |
| Instance-dependent, pure NE, informed | O(c' log T) | Ito et al. 2026 |
| Instance-dependent, pure NE, bandit | O(c log T) | Ito et al. 2026 |
| Instance-dependent, bandit | O(c₁ log T + √(c₂T)) | Ito et al. 2025 |
| Lower bound (adversarial bandits) | Ω(√(KT)) | Auer et al. 2002 |
| S=1 Markov game (model-based) | Õ(√(ABT)) | Bai & Jin 2020 |

For our 2×2 games (K=2), the worst-case bound is Õ(√(4T)) = Õ(√T),
which is essentially the standard bandit rate. The instance-dependent
O(log T) bound applies when there is a unique optimal pure strategy
with a positive gap — which is the case for Newcomb's problem (two-box
is strictly dominated under the true game).

### 9.2 Regret Definition Clarification

The literature uses several regret definitions:

- **Nash regret** (a.k.a. minimax regret): N·V(P) - Σ rewards, where
  V(P) is the minimax game value. This is our definition.
- **Pure-strategy maximin regret (PSMR)**: compares against the best
  pure maximin strategy. Coincides with Nash regret when the NE is in
  pure strategies.
- **External regret**: max_a Σ(reward_a - reward_t). Standard bandit
  regret. Weaker benchmark than Nash regret in games.

Our definition (Nash regret) is the standard one for this literature.

### 9.3 Updated Bound for Simplified IUCB

Based on the literature, our simplified algorithm should achieve:

- **Worst case**: Õ(K√(KT)) = Õ(K · √(KT)) Nash regret, or
  equivalently Õ(√(K²T)) following O'Donoghue et al.
- **Instance-dependent** (with gap Δ > 0): O(K² log(T) / Δ)
- **2×2 games**: Õ(√T) worst case, O(log T / Δ) with gap

These are tighter than our earlier conjectured O(K√(NT ln N)) bound
in Section 6.2 and dramatically better than Kosoy's K^{17/3} constants.

---

## 10. Recommendation: Implement As Planned

### 10.1 Summary of Findings

1. **Our approach is known and well-studied.** The "UCB for games"
   algorithm we independently derived is essentially the UCB algorithm
   from O'Donoghue et al. (2021) and the Maximin-UCB from Ito et al.
   (2026). This is reassuring — we are not reinventing something
   untested, but implementing a proven technique.

2. **The regret bounds are established.** Õ(√(K²T)) worst-case Nash
   regret is known, with O(log T) instance-dependent bounds for games
   with pure-strategy NE and positive gaps.

3. **No better algorithm exists for our specific needs.** More
   sophisticated algorithms (Tsallis-INF, GP-MW, OTN) offer marginal
   improvements in specific regimes but at the cost of significant
   complexity. For our 2×2 games with full outcome observation, the
   simple UCB approach is near-optimal and maximally transparent.

4. **Thompson Sampling fails.** O'Donoghue et al. (2021) show this
   explicitly — an important negative result that validates our choice
   of the optimistic (UCB) approach over a Bayesian one.

### 10.2 Decision: Implement as planned, with minor adjustments

We should implement our simplified IUCB exactly as specified in
Sections 2–4, with these minor adjustments informed by the literature:

1. **Name**: Consider calling it `MatrixUCBAgent` or `MaximinUCBAgent`
   rather than `MatrixUCBAgent`, to align with the established
   terminology from O'Donoghue et al. and Ito et al. This makes the
   algorithm's lineage clear and avoids confusion with Kosoy's IUCB.

2. **Cite the regret bound**: The Õ(√(K²T)) worst-case bound from
   O'Donoghue et al. (2021) applies directly. We do not need to
   reprove it.

3. **No design changes needed**: The algorithm as specified in Section
   2 is O'Donoghue et al.'s Algorithm 1 (with a standard UCB1 bonus
   term variant). The per-cell Hoeffding intervals, optimistic matrix
   construction, and minimax strategy computation match the published
   algorithm exactly. See Section 2.2 for the detailed comparison.

### 10.3 What We Should NOT Do

- **Do not adopt EXP3**: It doesn't exploit the matrix structure and
  performs worse empirically (O'Donoghue et al. 2021).
- **Do not adopt Thompson Sampling**: It fails catastrophically in
  matrix games (O'Donoghue et al. 2021).
- **Do not adopt GP-MW**: Overkill for 2×2 games; the Gaussian process
  overhead is unnecessary when K² = 4 cells.
- **Do not adopt Tsallis-INF**: While it has superior theoretical
  properties (best-of-both-worlds), it is significantly more complex
  and the marginal improvement on 2×2 games is negligible.
- **Do not use Kosoy's full IUCB**: The literature confirms that the
  general framework's overhead is unnecessary for payoff matrix games.

### 10.4 One Subtlety: Predictor Observes Agent's Strategy

Our setting has one feature that most of the literature does not
consider: the predictor observes the agent's mixed strategy before
choosing its action. In the standard matrix game learning setup, the
opponent chooses simultaneously or without seeing the agent's
mixed strategy.

This does not invalidate our approach because:
- Our regret benchmark is the *minimax value* V(P), which is the best
  the agent can guarantee against ANY opponent strategy, including one
  that observes the agent's mixed strategy.
- The confidence intervals are valid regardless of the predictor's
  strategy (the predictor controls which cells get observed but not
  the reward distribution within a cell).
- The optimism principle still applies: V(P_upper) >= V(P) with high
  probability.

In fact, the predictor observing our strategy makes the setting
*easier* in one sense: if the predictor plays a best response to our
strategy, then the per-round reward is exactly the game value of
P_upper against a best-responding opponent, which is exactly what
the minimax strategy is designed for. The regret decomposition in
O'Donoghue et al. (2021) applies directly.

---

## 11. What Does Kosoy's Framework Add Beyond the Prior Literature?

Given that our simplified algorithm matches existing published
approaches (§8.1), a natural question is: what does Kosoy's general
imprecise bandit framework (IUCB) contribute beyond what O'Donoghue
et al. (2021) and Ito et al. (2026) already provide?

### 11.1 For Payoff Matrix Games: Nothing Practical

For our specific application — 2×2 payoff matrix games — Kosoy's
IUCB solves the **exact same problem** as standard matrix game UCB,
with strictly worse constants.

In Example 4.3 of the thesis, the hypothesis θ IS the payoff matrix
P ∈ [-1,1]^4. For a known θ and agent strategy x, the credal set
K_θ(x) collapses: the predictor picks b adversarially, then the
reward is the deterministic value P[b,a]. There is no residual
imprecision within a cell. The lower prevision is just
min_b Σ_a x[a]·P[b,a] — the standard minimax game value.

The "imprecision" in this case is purely epistemic (we don't know P
yet), not genuine Knightian uncertainty. The entire apparatus — Z =
R^10, W = R^6, bilinear maps F, slab constraints, custom norms,
cycle-based updates — solves the same problem that per-cell Hoeffding
intervals + minimax solve, just embedded in a 10-dimensional space
for a 4-dimensional problem. The regret constants scale as K^{17/3}
versus Õ(K²) for the direct approach.

### 11.2 The Actual Contribution: Genuine Imprecision

The value of Kosoy's framework lies in its **generalization to
settings with genuine imprecision** — cases where even knowing the
true hypothesis θ does not pin down the outcome distribution. These
are cases the standard matrix game literature cannot handle:

**Adversary has hidden actions.** If the predictor can choose not
just an observable action b but also a noise distribution within
cell (b,a), the reward isn't the fixed value P[b,a] but drawn from
some adversarially-chosen distribution. Standard UCB assumes a fixed
cell mean; Kosoy allows the adversary additional within-cell freedom.

**Non-decomposable credal sets.** The uncertainty might not factor
as "adversary picks a row." It could be a convex polytope of joint
outcome distributions with complex constraints. The abstract bilinear
map F and slab-based confidence geometry handle this generality.

**Partial nonrealizability.** The credal set formulation means the
hypothesis θ specifies a *set* of outcome distributions, not a single
one. This gives a natural way to express "the environment is one of
these, but I don't know which." The regret guarantee is against the
lower prevision (worst-case over the credal set), providing
robustness to adversarial selection within the set. However, this is
partial — the true environment must still be *covered* by some
hypothesis's credal set.

**None of these apply to our payoff matrix games**, where credal sets
are degenerate (point distributions given (b,a)).

### 11.3 Decision-Theoretic Foundations

The deeper alignment motivation is connecting bandits to **decision
theory under Knightian uncertainty**:

- **Standard Bayesian bandits** (Thompson Sampling): place a prior
  over environments, compute posterior, sample. Fails if the prior
  is wrong — and O'Donoghue et al. (2021) showed Thompson Sampling
  fails *catastrophically* for matrix games.
- **Adversarial bandits** (EXP3): make no assumptions, get worst-case
  guarantees. Robust but doesn't exploit structure.
- **Kosoy's imprecise bandits**: maintain a *confidence set* of
  plausible environments (frequentist, not Bayesian), optimize for
  the worst case within the set (maximin / lower prevision). This is
  the decision-theoretic position known as **maximin expected utility**
  (Γ-maximin), connecting to the infra-Bayesian program.

The philosophical claim: for aligned AI, agents should use credal sets
rather than precise priors, because precise priors can be
catastrophically wrong, while credal sets express genuine uncertainty
and maximin over them gives robust guarantees.

### 11.4 Stepping Stone to Infra-Bayesian RL

The bandits paper is explicitly a building block for the larger
infra-Bayesian program (Kosoy's LessWrong/Alignment Forum sequence,
the `norabelrose/infrabayes` repo). The eventual goal is to extend
from bandits (no state) to full MDPs (sequential decisions), handle
temporal composition of credal sets (sa-measures), and provide regret
bounds for RL with genuine imprecision. The matrix game case is a
proof-of-concept for the simplest nontrivial case.

### 11.5 What About Irreflexivity / Newcomb Specifically?

Neither Kosoy's IUCB nor standard matrix game UCB addresses the core
decision-theoretic puzzle of Newcomb-like problems. Both are CDT
(Causal Decision Theory): they treat the predictor's action as
causally independent of the agent's strategy. The minimax value V(P)
is the CDT answer.

As noted in `20260320_ibplan.md` Section 17.5:
- **Newcomb**: both give the CDT answer (two-box, value = 5), not the
  EDT answer (one-box, value = 10).
- **Damascus**: CDT and EDT agree (p = 0.5), so both find the right
  answer.
- **Cooperation games**: both find the minimax equilibrium, not the
  cooperative one.

Exploiting the correlation between strategy and outcome (the
predictor observes the agent's mixed strategy) requires EDT/FDT-style
reasoning: "if I'm the kind of agent that one-boxes, the predictor
will predict that." Neither framework can express this. This is an
orthogonal problem to the one Kosoy's framework solves.

---

## 12. Nonrealizability: Kosoy vs. EXP3

A natural follow-up: if robustness to model misspecification is the
goal, why not just use EXP3? Adversarial bandit algorithms assume
*nothing* about the data-generating process, making them maximally
robust to nonrealizability.

### 12.1 The Guarantees

**EXP3**: O(√(KT log K)) regret against the best *fixed action* in
hindsight, with zero assumptions. The adversary can be adaptive, can
see the agent's mixed strategy, can change the payoff matrix every
round. Nothing breaks.

**Kosoy / Matrix UCB**: O~(√(K²T)) regret against the *minimax game
value*, assuming: (1) a fixed payoff matrix (stationarity), (2)
bounded rewards, (3) correct observation model, (4) the environment
is covered by the hypothesis class.

**EXP3 is strictly more robust to nonrealizability.** If any of
Kosoy's assumptions fail — non-stationary payoffs, hidden state,
corrupted observations — her guarantee is void. EXP3 still works.

### 12.2 The Benchmarks Are Different

But the robustness comes at the cost of a much weaker benchmark.

On **Damascus** (P = [[0,10],[10,0]]):
- Best fixed action: action 0 or 1, both get reward 0 (Death
  predicts correctly). EXP3's benchmark is **0**.
- Minimax value: p = 0.5 gets expected reward 5. UCB/IUCB's
  benchmark is **5**.

EXP3 can have zero regret while getting reward 0 every round. It
"wins" by not doing worse than the best pure strategy, but it
**never discovers that randomizing is better**. It cannot compete
against mixed-strategy benchmarks because its regret definition
doesn't account for game structure.

Matrix UCB discovers the mixed equilibrium because it competes
against a benchmark that values mixed strategies.

### 12.3 The Fundamental Tradeoff

There is a fundamental tradeoff between the **strength of
assumptions** and the **strength of guarantees**:

| | Assumptions | Benchmark | Robustness |
|---|---|---|---|
| **EXP3** | None | Best fixed action | Maximally robust |
| **Matrix UCB** | Fixed payoff matrix | Minimax value | Moderate |
| **Kosoy IUCB** | Linear hypothesis class, credal sets correct | Lower prevision | Moderate |

Each step up in assumptions buys a stronger benchmark but sacrifices
robustness. EXP3 assumes nothing but guarantees little. Kosoy assumes
the most but (in the general case) guarantees the most.

### 12.4 Specific Failure Modes

| Type of failure | EXP3 | Kosoy / Matrix UCB |
|---|---|---|
| Adversarial / non-stationary rewards | Handles it (core design) | Breaks (assumes fixed P) |
| Unknown reward range | Handles it | Breaks (needs bounded range) |
| Payoff matrix structure is wrong | Handles it (doesn't assume it) | Breaks |
| Hidden state / memory | Handles it | Breaks |
| Predictor has more freedom than modeled | Handles it | Breaks |
| **Exploiting game structure** | Cannot (best fixed action) | Can (mixed strategies) |
| **Stronger benchmark** | No (best pure action) | Yes (minimax value) |

### 12.5 The Open Question: Best of Both Worlds for Games

The interesting question for alignment: **is there an algorithm that
exploits game structure when it exists but degrades gracefully when
the structure is wrong?**

In standard bandits, this question is answered: Zimmert & Seldin's
Tsallis-INF (2019/2021) achieves optimal O(√(KT)) adversarial regret
AND O(Σ log(T)/Δᵢ) stochastic regret simultaneously. It adapts to
the difficulty of the environment without knowing in advance whether
the setting is stochastic or adversarial.

For matrix games, the analog would be an algorithm that:
- Competes against the minimax value when payoffs are stationary
  (exploiting game structure, playing mixed strategies)
- Degrades to best-fixed-action regret when payoffs are
  non-stationary or the game structure is wrong
- Detects when its structural assumptions are violated

Ito et al. (2025) use Tsallis-INF for matrix games and achieve
instance-dependent bounds that adapt to game difficulty. But their
analysis still assumes a fixed payoff matrix. Whether the
best-of-both-worlds property extends to the game-structure-vs-no-
structure distinction appears to be open. This is perhaps the most
alignment-relevant open question in this space: how to get robust
guarantees (EXP3-like) while still exploiting structure (UCB-like)
when it exists.

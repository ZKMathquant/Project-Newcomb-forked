# Literature Review Plan: Learning in Unknown Payoff Matrix Games

**Date**: 2026-03-22
**Context**: We are implementing a bandit agent for Newcomb-like games
(2-action games with a predictor that observes the agent's mixed strategy).
We have a full implementation of Kosoy (2024) "Imprecise Multi-Armed
Bandits" (IUCB) but found it impractical (see `20260320_ibplan.md`
Sections 16-17). We designed a simplified "UCB for games" approach
(see `20260321_simpleIUCB.md`) that maintains per-cell confidence
intervals on the payoff matrix and plays optimistic minimax. Before
implementing, we want to know what already exists.

---

## 1. The Specific Problem Setting

An agent repeatedly plays a game with the following structure:

- Unknown K×K payoff matrix P, where P[b, a] = expected reward when
  predictor plays b and agent plays a
- Each round: agent declares mixed strategy x, predictor observes x
  and chooses action b (strategy unknown — could be adversarial,
  cooperative, or anything), agent samples action a ~ x
- Agent observes the FULL outcome: (b, a, reward)
- Goal: minimize minimax regret = N*V(P) - Σ rewards, where
  V(P) = max_x min_b Σ_a x[a]*P[b,a]
- Rewards are bounded (known range) and may be stochastic given (b,a)

Key features that distinguish this from generic bandits:
- The payoff depends on BOTH the agent's action and the predictor's
  action (game structure, not independent arms)
- The agent observes the predictor's action (not just the reward)
- The agent plays MIXED strategies (not deterministic)
- The predictor can see the agent's mixed strategy before choosing

## 2. Questions to Answer

### 2.1 Existing Algorithms
- Has "optimistic minimax with per-cell confidence intervals" been
  studied? Under what name?
- What is the state of the art for this problem setting?
- Are there algorithms with proven regret bounds for learning in
  unknown matrix games with full outcome observation?

### 2.2 Regret Bounds
- What are the best known regret bounds for this setting?
- Do they depend on K as K, K², or worse?
- Is there a matching lower bound (minimax-optimal rate)?
- Does the cooperative vs adversarial predictor distinction matter
  for the bounds?

### 2.3 Related Settings
- **Bandit feedback** (agent only observes reward, not predictor's
  action): How does this change the problem? Relevant if we ever
  lose access to env_action.
- **Online matrix games** (adversarial rewards, no fixed P): How do
  no-regret algorithms like EXP3 compare? When is exploiting the
  payoff matrix structure worth it?
- **Stochastic games / Markov games**: Our setting is the single-state
  (bandit) special case. What's known for this special case?
- **Thompson Sampling for games**: Bayesian counterpart to our
  frequentist approach. Existing results?

### 2.4 Relationship to Kosoy (2024)
- Does the literature on matrix game learning subsume the payoff
  matrix special case of Kosoy's imprecise bandits?
- Are there results that give tighter bounds for this special case
  than Kosoy's general K^{17/3} constant?
- Is there work that explicitly connects imprecise probability /
  credal sets to game-theoretic learning?

### 2.5 Practical Algorithms
- What algorithms are actually used in practice for similar problems
  (e.g., security games, poker with unknown opponents)?
- Are there implementations we can reference or reuse?

## 3. Search Strategy

### 3.1 Key Terms to Search

Primary:
- "learning unknown matrix game"
- "optimistic minimax" + bandits
- "UCB matrix game"
- "stochastic game bandit"
- "game-theoretic exploration exploitation"

Secondary:
- "UCRL single state game"
- "R-MAX matrix game"
- "Thompson sampling zero-sum game"
- "regret minimization unknown payoff matrix"
- "bandit learning Nash equilibrium"

Related:
- "counterfactual regret minimization unknown game"
- "online learning game theory survey"
- "imprecise probability bandit"
- "credal set learning"

### 3.2 Key Authors / Groups

- The online learning / game theory group: Cesa-Bianchi, Lugosi
- Bandits: Lattimore, Szepesvári (their "Bandit Algorithms" book
  likely covers game-theoretic extensions)
- Game-theoretic learning: Fudenberg, Levine
- Imprecise probability: Kosoy, Garrabrant (MIRI-adjacent)
- UCRL / optimistic RL: Jaksch, Ortner, Auer

### 3.3 Key Venues

- COLT, NeurIPS, ICML, ALT (theory conferences)
- Journal of Machine Learning Research
- Games and Economic Behavior
- arXiv cs.LG, cs.GT, stat.ML

## 4. Deliverables

After the review, update `20260321_simpleIUCB.md` with:

1. A new section listing the most relevant prior work, with citations, hyperlinks,
   and brief descriptions of how each relates to our setting
2. Updated regret bound analysis incorporating known results
3. A recommendation: implement our simplified approach as planned,
   adopt an existing algorithm, or adapt one
4. Any changes to the algorithm design informed by the literature

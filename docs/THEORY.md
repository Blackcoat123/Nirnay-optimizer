# NIRNAY — the theory

What NIRNAY computes, how, and why each method is the one used. This document is about the
mathematics, not the code: where a method lives in the source, the file is named once so a
reader can find it, and [`ARCHITECTURE.md`](ARCHITECTURE.md) is the map of the code itself.
Every method here is implemented from the literature cited beside it
([`PROVENANCE.md`](PROVENANCE.md) holds the full table); no other solver's source was read.

*Nirṇaya* (निर्णय) is Sanskrit and Hindi for *decision*. An optimization solver is a decision
engine: given what a plant can do and what things cost, it decides what to do.

---

## 1. The problems

### 1.1 Linear programming (LP)

NIRNAY keeps every model in one two-sided form, the form MPS files already describe:

$$
\min_x \; c^\top x \quad \text{subject to} \quad r_L \le A x \le r_U, \qquad l \le x \le u,
$$

with $A$ an $m \times n$ sparse matrix and any bound allowed to be infinite. An equality row
has $r_L = r_U$; a "≤" row has $r_L = -\infty$; a range row has both finite. Maximisation is
handled by negating $c$ internally and flipping the signs back on report, so every reported
dual keeps the sign convention of the file the user wrote.

Keeping the two-sided form — rather than converting to $Ax = b,\ x \ge 0$ with slack and
split variables — matters: the conversion doubles range rows and free variables, changes
the geometry every method sees, and adds a transformation that has to be undone exactly.

### 1.2 Duality: what the "other half" of an answer is

Every LP has a dual. With multipliers $y$ on the rows and reduced costs $d$ on the columns,

$$
d = c - A^\top y ,
$$

the point $x$ is optimal exactly when there is a $y$ such that (the **KKT conditions**):

1. **primal feasibility** — $x$ satisfies every row and every bound;
2. **dual feasibility** — each $y_i$ and $d_j$ has the sign its bound allows (a positive
   reduced cost is only allowed where $x_j$ can sit at a finite lower bound, and so on);
3. **complementary slackness** — a multiplier is non-zero only on a constraint that is tight.

When all three hold, the primal objective $c^\top x$ equals the dual objective (the
Lagrangian bound), and that equality is a *proof* of optimality anyone can check without
trusting the solver. This is why NIRNAY reports the whole primal–dual pair and why its
independent verifier re-checks all three conditions.

The dual variables are not a mathematical by-product; they are the answer to the questions a
planner asks. $y_i$ is the **shadow price** of row $i$ — what one more unit of that
constraint's right-hand side is worth: one more tonne of crude-unit capacity, one more ppm of
sulphur allowance, one more barrel of diesel commitment. $d_j$ is how much the price of
activity $j$ would have to change before it enters the plan.

### 1.3 Proofs for the answers that are not a point

- **Infeasible.** Farkas' lemma: the rows and bounds have no common point exactly when some
  non-negative combination of them yields a contradiction like $0 \ge 1$. NIRNAY returns
  that combination — the *Farkas vector* — and the verifier checks the contradiction.
- **Unbounded.** A feasible point plus a *ray* $r$ with $A r$ within the row directions,
  $r$ within the bound directions, and $c^\top r < 0$: the objective improves without limit
  along it. Both are returned and checked.

### 1.4 Mixed-integer (MILP)

The same model with some columns restricted to integers. It is no longer convex, so there
is no dual certificate; optimality is proved by **bounding** — showing that no part of the
search space can hold anything better than the incumbent (section 9).

### 1.5 Quadratic (QP, MIQP)

The objective gains $\tfrac12 x^\top Q x$ with $Q$ symmetric. When $Q$ is positive
semidefinite the problem is convex and every local optimum is global. When it is not, a local
optimum is not global, and reporting one as "optimal" would be wrong in a way nobody could
see. NIRNAY therefore **decides convexity before solving** (section 10) and refuses a
non-convex QP with a certificate rather than approximating it.

---

## 2. The pipeline

```
 read (MPS / QPS / LP)  →  classify (LP, MILP, QP, MIQP)  →  presolve  →  scale
      →  engine(s)  →  finish (polish, crossover, re-solve)  →  unscale  →  postsolve
      →  measure the answer against the ORIGINAL model  →  status guard  →  report
```

Two principles run through all of it:

- **The model the user wrote is the one that is judged.** Presolve, scaling and the engines
  work on transformed problems; the answer is mapped back and every claim — feasible,
  optimal — is re-measured on the original. An engine that believes it has converged but
  whose point does not measure within tolerance is downgraded, not believed.
- **A status is a claim with evidence attached.** `optimal` comes with a dual certificate,
  `infeasible` with a Farkas vector, `unbounded` with a ray; `feasible` means a point that
  satisfies every constraint but whose optimality is not proved; a limit status says the run
  was cut short and hands back its best point as exactly that.

---

## 3. Presolve

Presolve removes what can be decided without solving: empty rows and columns, fixed
columns, singleton rows (which are really bounds), redundant rows (implied by the bounds),
forcing rows (whose bounds force every variable in them to a bound), free-column singletons
and doubleton equations (a variable that can be substituted out). Each reduction is pushed on
a stack.

Postsolve pops the stack in reverse and rebuilds not only $x$ but the **duals** of the
original model — the part most presolvers get subtly wrong, because a removed row's multiplier
has to be reconstructed from the reduced costs it absorbed. NIRNAY runs the dual passes to a
fixed point and then recomputes every touched reduced cost as $c - A^\top y$, so a postsolve
error shows up as a measurable violation rather than a quietly wrong price.

---

## 4. Scaling (equilibration)

Industrial models mix units: flows in tonnes, qualities in parts per million, costs in
crores. The matrix entries then span ten or more orders of magnitude, which slows every
method and destabilises factorizations. NIRNAY solves the scaled problem
$\hat A = D_r A D_c$ with diagonal $D_r, D_c$ chosen by

- **Ruiz equilibration** (Ruiz 2001): ten rounds of dividing each row and column by the
  square root of its largest entry, which drives every row and column infinity norm
  towards 1; then
- **Pock–Chambolle** diagonal preconditioning with $\alpha = 1$ (Pock & Chambolle 2011):
  rows and columns scaled by the square roots of their absolute sums, which is what the
  step-size theory of the first-order method is stated in.

Scaling changes the numbers, never the answer: $x = D_c \hat x$, $y = D_r \hat y$, and the
answer is always measured unscaled.

---

## 5. The simplex method

### 5.1 Vertices and bases

The feasible region of an LP is a polyhedron, and if an optimum exists one is at a vertex. A
vertex is described by a **basis**: $m$ of the $n + m$ variables (structurals plus one
logical per row) are *basic* and determined by the equations; the rest are *nonbasic* and sit
at a bound. Moving to an adjacent vertex swaps one basic and one nonbasic variable — a
**pivot**. The simplex method moves from vertex to vertex, each pivot not worsening the
objective, until no improving direction exists.

The *revised* simplex never forms the full tableau. It keeps a factorization of the basis
matrix $B$ and computes what it needs by solving with it: **FTRAN** ($B\,a = a_q$, the entering
column) and **BTRAN** ($B^\top y = c_B$, the prices). All variables are bounded-variable
aware, so a nonbasic variable can sit at either bound and a "bound flip" is a pivot that
changes no basis.

### 5.2 Primal and dual simplex

- The **primal simplex** keeps the point feasible and works towards dual feasibility
  (optimality). Phase 1 first finds a feasible point, here with a composite objective rather
  than a big-M.
- The **dual simplex** keeps the reduced costs dual feasible and works towards primal
  feasibility. It is NIRNAY's default LP engine because it is the natural engine after a
  change of bounds — which is exactly what branch and bound does at every node — and because
  it was measured to solve more of Netlib in less time. NIRNAY's dual simplex uses the
  **bound-flipping ratio test** (long-step: pass over breakpoints where a boxed variable can
  flip bound instead of entering), temporary artificial bounds for columns that start dual
  infeasible (Koberstein 2005), and hands the basis to the primal simplex whenever it cannot
  finish honestly.

### 5.3 Pricing, ratio tests, degeneracy

- **Pricing** chooses the entering (primal) or leaving (dual) variable. **Devex**
  (Forrest & Goldfarb 1992) approximates steepest edge with reference weights; it cut
  iterations by a third against Dantzig's largest-coefficient rule on the Netlib medium tier.
- **Ratio tests** choose how far to move. The textbook test is exact; Harris's two-pass test
  trades a tolerance for a larger, better-conditioned pivot.
- **Degeneracy** — many constraints tight at one vertex — makes the simplex pivot without
  moving (stalling, and in theory cycling). NIRNAY perturbs bounds or costs by tiny random
  amounts when it detects a stall, and removes the perturbation before declaring optimality.

### 5.4 The linear algebra underneath

- **Sparse LU with Markowitz threshold pivoting.** $B$ is factorized choosing pivots that
  minimise fill-in (Markowitz's count $(r-1)(c-1)$), subject to a threshold that keeps each
  pivot at least 1% of its column's largest entry, for stability.
- **Updates.** After each pivot the factorization is updated in product form (an eta file)
  instead of refactorized; it is refactorized when the accumulated update work exceeds a
  measured multiple of the factor's size, and whenever an FTRAN residual check shows the
  update has lost accuracy.
- **Hyper-sparsity.** When the right-hand side of a solve is very sparse, so usually is the
  result; NIRNAY's FTRAN traverses only the reachable part of $U$ (Gilbert–Peierls style),
  so a solve costs in proportion to its output rather than to $m$.
- **Basis repair.** A basis that turns out singular has its dependent columns replaced by
  slacks rather than aborting the solve.
- **Iterative refinement** (Wilkinson 1963) of the final basic solution, primal and dual,
  with compensated residuals.
- **Optimality only on fresh factors.** The final optimality check is always made after a
  fresh factorization, never on an updated one.

---

## 6. The interior-point method

### 6.1 The central path

Instead of walking the boundary, an interior-point method walks through the inside of the
feasible region. It replaces "$x \ge l$" by a barrier $-\mu \sum \log(x - l)$ and follows the
minimisers of the barrier problem as $\mu \to 0$ — the **central path** — to the optimum. In
primal–dual form this is Newton's method on the KKT conditions with complementarity relaxed
to $x_j z_j = \mu$.

### 6.2 Mehrotra's predictor–corrector

Each iteration takes an affine-scaling ("predictor") step that ignores centrality, uses how
far it could go to choose how much centring is needed, then takes a combined corrector step
with a second-order term (Mehrotra 1992; Wright 1997). In practice 20–60 iterations reach
$10^{-8}$ on almost any LP, regardless of size — each iteration is the expensive part.

### 6.3 The linear algebra

Every Newton step reduces to the **normal equations**

$$
(\bar A\, \Theta\, \bar A^\top + \delta I)\, \Delta y = r ,
$$

with $\Theta$ diagonal and changing every iteration. That matrix is symmetric positive
definite, so it is factorized as $L D L^\top$. The order in which rows are eliminated decides
how much fill the factor gets; NIRNAY computes it with **approximate minimum degree** on the
quotient graph (Amestoy, Davis & Duff 1996), once, and refactorizes numerically each
iteration. A small primal–dual regularization (Altman & Gondzio 1999) keeps the pivots away
from zero as the method converges and $\Theta$ becomes extremely ill-conditioned.

### 6.4 Trade-offs

The interior point needs few iterations and each is a factorization — so it wins on models
whose factor stays sparse (structured, multi-period planning models) and loses on models whose
factor fills in (random or expander-like structure). It returns a point on the optimal face,
not a vertex, and no basis — which is what crossover (section 8.2) repairs.

---

## 7. The first-order method: restarted PDHG

### 7.1 Why a first-order method at all

The simplex is sequential — every pivot depends on the one before — and the interior point is
dominated by a sparse factorization, which parallelises poorly and whose memory grows with
fill. Neither maps onto a GPU. A **first-order method** uses only matrix–vector products with
$A$ and $A^\top$: no factorization, memory linear in the model, no serial dependency inside an
iteration. That is the method that scales to millions of variables and the method that runs
on a GPU; it is how the field has approached very large LPs since 2021 (Applegate et al.;
Lu & Yang).

### 7.2 The saddle point

The LP is written as a saddle-point problem between $x$ and the row multipliers $y$:

$$
\min_{x \in X} \max_y \; c^\top x + y^\top A x - \sigma_C(y), \qquad X = [l, u],\ C = [r_L, r_U],
$$

where $\sigma_C$ is the support function of the row-bound box. The inner max over $y$
reproduces the constraints: any violation of $r_L \le Ax \le r_U$ lets $y$ drive the
objective to infinity.

### 7.3 The PDHG iteration

The primal–dual hybrid gradient method (Chambolle & Pock 2011) alternates a projected
gradient step in $x$ with a proximal step in $y$ at the **extrapolated** point:

$$
x^{+} = \operatorname{proj}_X\!\big(x - \tau (c + A^\top y)\big), \qquad
y^{+} = \operatorname{prox}_{\sigma\,\sigma_C}\!\big(y + \sigma A (2x^{+} - x)\big).
$$

Moreau's identity turns the $y$ step into a projection onto the row-bound box, which handles
equality, one-sided, range and free rows with one expression. The step sizes satisfy
$\tau \sigma \|A\|_2^2 < 1$; NIRNAY sets $\tau = \eta/\omega$, $\sigma = \eta\omega$ with
$\eta$ just below $1/\|A\|_2$ (estimated by power iteration on $A^\top A$) and $\omega$ the
**primal weight** that balances progress in $x$ against progress in $y$.

One iteration is two sparse products, $A x^{+}$ and $A^\top y^{+}$, plus element-wise work.
Because the products are linear, NIRNAY carries each iterate together with its products and
updates them by linearity wherever the algorithm combines iterates, so an iteration never
needs a third product and a convergence check needs none.

### 7.4 Restarts

Plain PDHG converges at a sublinear rate. LPs, however, are *sharp* — the distance to the
optimal set is bounded by a multiple of the KKT error — and on sharp problems **restarting**
the method from a good recent point turns sublinear convergence into linear convergence
(Applegate, Hinder, Lu & Lubin 2021). NIRNAY restarts when the error has dropped by a fixed
factor since the last restart (sufficient decay), when it has dropped somewhat but stopped
improving (necessary decay without progress), or after a fixed fraction of the run
(artificial restart), and moves the primal weight towards the observed ratio of dual to
primal movement at each restart (PDLP §3.2).

### 7.5 Two schemes

- **Adaptive (PDLP).** The step size is adapted every iteration from the observed ratio of
  movement to interaction, and restarts go to whichever of the current iterate and the
  running average has the smaller KKT error (Applegate et al. 2021). NIRNAY's earlier engine
  claimed to do this but always restarted at the current iterate; fixing that alone cut
  iterations 10–26× on the committed Netlib instances.
- **Reflected restarted Halpern PDHG — the default.** Write one PDHG step as an operator
  $z \mapsto T(z)$ on $z = (x, y)$. $T$ is firmly nonexpansive in the metric
  $M = \begin{bmatrix} I/\tau & -A^\top \\ -A & I/\sigma \end{bmatrix}$, so its fixed points are
  the saddle points. **Halpern's iteration** (Halpern 1967) anchors every step back towards the
  last restart point $z^0$:

  $$ z^{k+1} = \tfrac{k+1}{k+2}\,\big((1+\rho)\,T(z^k) - \rho\, z^k\big) + \tfrac{1}{k+2}\, z^0 , $$

  which gives the optimal $O(1/k)$ rate on the **fixed-point residual** $\|z - T(z)\|_M$;
  the reflection $\rho \in [0,1]$ applies the anchoring to the nonexpansive operator
  $(1+\rho)T - \rho I$, which takes longer steps (Lu & Yang 2024). Restarts are triggered by
  the decay of that residual, and the step size is constant. The measured effect: 0.36–0.66×
  the adaptive scheme's iterations on every committed Netlib instance, all nine to optimal
  with no finishing step, and 2,496 against 4,160 iterations on the 365-period refinery
  model. The constant step also means a block of iterations needs **no decision at all** in
  between — which is what lets the GPU run them without a single transfer.

A divergence guard halves the step and restarts from the best point should the residual
ever blow up (the norm estimate approaches $\|A\|$ from below before it is rounded up).

### 7.6 When to stop, and what "optimal" means

The loop measures, every 64 iterations, the relative primal residual, relative dual residual
and relative duality gap **in the original units** (never the scaled ones, which a good
preconditioner could flatter), plus the absolute violations and complementarity the
independent verifier uses. It stops only on a point that meets both the requested relative
tolerance *and* the project's absolute standard ($10^{-7}$ primal and dual, $10^{-9}$ relative
gap) — and before it stops, the point is copied to the host, unscaled, and re-measured with
fresh products against the original model. `optimal` from PDHG therefore means exactly what
it means from the simplex: a point the verifier accepts.

### 7.7 On a GPU

A PDHG iteration is **memory-bound**: it streams the matrix (twice) and a dozen vectors
through the processor and does about one multiply–add per byte loaded. A B200's HBM3e
delivers about 8 TB/s against a server CPU's ~0.3 TB/s, so the GPU's advantage is
bandwidth, and it shows exactly where the model is large enough to stream. NIRNAY's CUDA
backend:

- keeps every vector — iterate, output, anchor, best point, running average, and each one's
  two products — **resident on the device for the whole solve**;
- stores $\hat A$ and $\hat A^\top$ as two row-compressed matrices, so both products are
  row-parallel gathers (cuSPARSE, NVIDIA's vendor sparse BLAS, with its deterministic CSR
  algorithm);
- fuses each half-step's projection, extrapolation, Halpern update and the reductions it
  needs into single kernels;
- performs every reduction in two passes with a fixed block tree and a fixed grid, so a run
  on a given device is reproducible bit for bit;
- sends to the host only scalars — three per block of 64 iterations, eight per convergence
  check — and whole vectors only for the final answer and its host re-measurement;
- stays in double precision throughout.

---
## 8. Finishing a first-order answer

A first-order method converges linearly to a *tolerance*; its answer is a point near the
optimal face with near-optimal duals. Three finishing steps turn it into more.

### 8.1 Interior-point polish

When PDHG stops short of the standard (at a limit), its point, row duals and reduced costs
become the starting point of the interior point with a small iteration budget. A
second-order method started near the optimum converges quadratically, so a few iterations
take a $10^{-4}$ point to $10^{-10}$ — provided its factor is affordable, which it measures
from the ordering before building anything, and declines within a time budget if not.

### 8.2 Crossover to a basis

A refinery planner wants a *vertex* — an operating plan with exact shadow prices — and branch
and bound wants a basis to warm-start from. Crossover recovers one: an optimal basis can be
recovered from an optimal primal–dual pair in strongly polynomial time (Megiddo 1991), and in
practice by **identifying** the basis the point sits on and letting the simplex **clean up**
(Bixby & Saltzman 1994). NIRNAY ranks every variable by an indicator (El-Bakry, Tapia & Zhang
1994) — how far its value is from its nearest bound against how large its reduced cost or
price is — takes the $m$ most "basic-looking" as the basis, places the rest at their nearest
bounds, and runs the dual simplex from there. It replaces the first-order answer only with an
**optimal** simplex answer, so it can never make a result worse. On small and medium models it
costs milliseconds and lands on the simplex's objective to the last digit; at scale the dual
simplex's iteration rate is the limit, so by default it runs where it pays.

### 8.3 Re-solving an unproven optimum from its own basis

The simplex declares optimality on its scaled working problem; the status guard then
measures on the unscaled model, and on a badly scaled model the two can disagree by a small
factor. When that happens NIRNAY re-runs the dual simplex *warm-started from the basis it
finished on*, with internal tolerances a hundred times tighter, and keeps the result only if
it then measures within tolerance. On Netlib `pilot87` this turned a downgraded `feasible`
answer into the true optimum, 301.710347333, agreeing with HiGHS to $10^{-15}$ — and showing
that Netlib's published value (301.71072827) is off by $1.3\times10^{-6}$.

---

## 9. Concurrent optimization

No LP engine is best everywhere, and the measurements say exactly where each one wins: the
dual simplex takes most Netlib instances in milliseconds and cannot finish a 32,000-row
refinery year in five minutes; PDHG on a GPU solves that year in seconds and is slower than
the simplex on anything small; the interior point sits between them on structured models.
Choosing by a size threshold guesses; racing measures. `algorithm=concurrent` — and `auto` on
any model of 1,000 rows or more — runs the dual simplex, PDHG (on the GPU when there is one)
and the interior point in parallel threads. The first engine to return a **judged** optimal
answer (the status guard is applied inside the race, so an optimum that would be downgraded
cannot win) interrupts the others through a shared stop flag that reaches inside their
factorizations. The race rescued Netlib `maros-r7` (a numerical failure for the dual simplex,
optimal from PDHG in seconds) and `dfl001` (a time limit before).

---

## 10. Mixed-integer programming: branch and bound

### 10.1 The method

Drop integrality and solve the **LP relaxation**: its optimum is a bound no integer point can
beat. If the relaxation's solution is integral, it is optimal. Otherwise pick a fractional
integer variable $x_j = v$ and **branch** into two subproblems, $x_j \le \lfloor v \rfloor$ and
$x_j \ge \lceil v \rceil$, which between them contain every integer point. Recurse. A
subproblem is **fathomed** when its relaxation is infeasible, integral, or — the one that
matters — when its bound is no better than the best integer point found so far (the
**incumbent**). The search ends when nothing is open (proved optimal) or when the gap between
the incumbent and the best open bound meets the requested tolerance.

NIRNAY's tree never copies the model: one working model exists, and a node is entered by
applying its chain of bound changes from the root and left by undoing them. The open nodes are
kept in a binary heap on their bounds, so choosing the next node is $O(\log n)$ — the list used
to be scanned in full every node, which was 76% of the solve time on small MIPLIB models.

### 10.2 The ingredients that decide performance

- **Node LPs are warm-started dual simplex solves.** A child differs from its parent by one
  bound, so the parent's optimal basis is still dual feasible — exactly what the dual simplex
  starts from. A node typically costs a handful of pivots.
- **Bound propagation** at every node (Savelsbergh 1994): row activity bounds tighten column
  bounds, often proving a node infeasible with no LP solve at all.
- **Node selection:** depth-first while diving (to find incumbents early), best-bound
  otherwise (to raise the global bound).
- **Branching — reliability branching** (Achterberg, Koch & Martin 2005): each candidate is
  scored by the product of its estimated bound gains in both directions. Estimates come from
  **pseudocosts** (the average observed gain per unit of fractionality) once a column has
  enough observations, and from **strong branching** (actually solving both children, with an
  iteration cap) until then.
- **Cutting planes** at the root (optional): Gomory mixed-integer cuts from the optimal tableau
  and lifted knapsack cover cuts, each checked for validity against exact rational arithmetic
  in the test suite; and integer rounding that tightens rows in place.
- **Primal heuristics**, because pruning needs a good incumbent:
  - *rounding* of every node's relaxation;
  - *diving* from the root — fix the least fractional variable, re-solve, repeat;
  - the **feasibility pump** (Fischetti, Glover & Lodi 2005; Bertacco, Fischetti & Lodi 2007)
    when the dive finds nothing: alternately round the LP point and project the rounding back
    onto the LP polyhedron in the L1 distance, flipping the most-disagreeing variables to
    escape cycles;
  - **RINS** (Danna, Rothberg & Le Pape 2005) once an incumbent exists: fix every integer
    variable on which the incumbent and the current node relaxation agree, add an objective
    cutoff, and solve the small remaining MIP with a node budget — at geometrically spaced
    nodes, so its share of the search stays bounded.

  Every candidate, from any heuristic, is re-checked for integrality and every row against the
  original model before it can become the incumbent.

### 10.3 What "optimal" means for a MIP

Either the tree was exhausted (the incumbent is proved optimal), or the relative gap between
the incumbent and the best open bound met `mip_relative_gap` (default $10^{-4}$) — optimal
*within the requested tolerance*, reported as such, with the achieved gap and the open bound
in the answer so the claim is exactly what was proved.

---

## 11. Quadratic programming

### 11.1 Deciding convexity first

$Q$ is factorized as $L D L^\top$ with symmetric pivoting; a negative pivot is a direction of
negative curvature and proves non-convexity. NIRNAY does this **before any arithmetic of the
solve**, refuses a non-convex model with that pivot as the certificate, and never reports a
local optimum as a global one.

### 11.2 The convex QP engine

A primal–dual splitting method (Condat 2013; Vũ 2013) — the QP counterpart of PDHG, whose
primal step is a gradient step on $c + Qx$ — converged to a tight tolerance and measured by the
same absolute standard.

### 11.3 MIQP

Branch and bound with the QP engine as the node solver. Because a first-order node bound is
only accurate to its tolerance, and an optimistic bound can fathom the subtree that holds the
optimum, the node tolerance is tightened well below the gap targets and the pruning margin is
widened by the same amount: the safe direction, paid for in nodes.

---

## 12. Numerical robustness

- **Degeneracy** — handled by perturbation and bound-flipping (section 5.3); the generated
  degenerate cases in the robustness suite are checked for the exact rank by the independent
  checker.
- **Ill-conditioning** — equilibration (section 4), threshold pivoting, refactorization on a
  measured accuracy loss, iterative refinement, regularized interior-point pivots. The
  robustness suite holds the optimum across a $10^{22}$ spread of matrix entries.
- **Measuring in the right units.** An absolute tolerance of $10^{-7}$ on a row whose terms are
  $10^{7}$ asks for accuracy below double precision's reach. Every violation is therefore
  judged relative to the magnitude of the terms it was computed from, and the absolute value is
  what is printed.
- **The status guard.** Every answer's status is reconciled with its measured quality: an
  "optimal" that violates feasibility becomes a numerical error; one that violates dual
  feasibility becomes "feasible"; an integer column that is fractional is a relaxation, not a
  MIP answer; a non-finite number is never published as a point.

---

## 13. How correctness is established

1. **An independent verifier** (`tools/verify_solution.py`) re-reads the model with its own
   parser and checks the written solution for bounds, rows, integrality, reduced costs, dual
   feasibility, complementary slackness and strong duality — or checks the Farkas vector or ray.
   It shares no code with the solver.
2. **An exact rational oracle** in the test suite: a rational-arithmetic simplex and an exact
   MILP branch and bound that the floating-point engines are fuzzed against on generated
   instances nobody chose.
3. **Optima known by construction.** The scale families are generated *backwards* from a
   primal–dual pair that already satisfies the KKT conditions, from integer data, so the
   optimum is known exactly before the solver sees the file — at a million variables, where no
   other reference exists.
4. **A second solver.** Every Netlib answer that disagrees with the published table is
   cross-checked against HiGHS run as a separate process. On every such instance NIRNAY's
   verified answer agrees with HiGHS; the published table is the outlier.
5. **GPU against CPU.** The two PDHG backends are pinned to each other and to the simplex in the
   test suite.

---

## 14. The industrial models

The benchmark sets test the engine; these test the claim that it serves refinery work.

- **Refinery planning over T periods** (`bench/runners/generate_refinery_lp.py`): crude
  purchases and distillation throughput per crude; production by fixed yields; product sales;
  crude and product tanks with inventory balances linking each period to the next; unit
  capacities; product quality specifications as linear blending budgets; delivery commitments.
  The operating plan is chosen first and the prices derived from the KKT conditions, so the
  optimum is exact by construction. T = 12 is a monthly year (1,068 rows), 365 a daily year
  (32,485 rows), 8,760 an hourly year (779,640 rows × 1,208,880 columns, 10 million nonzeros).
  The staircase structure — each period coupled only to the next — is what makes the interior
  point's factor stay sparse and PDHG's restarts effective.
- **Crude blending** (`demo/crude_blend.mps`, and a QP variant with price impact): crudes into a
  product pool under throughput, commitment and sulphur limits. Its shadow prices are the
  numbers a planner acts on — the marginal cost of the diesel commitment, the value of the
  sulphur specification.
- **Refinery scheduling** as a MILP (`demo/blend_milp.mps`), **power dispatch** as unit
  commitment, **production planning** as lot sizing, and **supply chain** distribution
  (`data/casestudies/`), each settled against an exhaustive or exact reference.

---

## 15. Glossary

- **Basis** — the $m$ variables determined by the equations at a vertex; the rest sit at bounds.
- **Crossover** — recovering a basis from an interior or first-order point.
- **Dual simplex** — keeps optimality conditions, restores feasibility; the node engine of branch
  and bound.
- **Fathom** — discard a branch-and-bound node that cannot improve on the incumbent.
- **Farkas vector** — a combination of constraints proving infeasibility.
- **Fixed-point residual** — $\|z - T(z)\|$; zero exactly at a saddle point.
- **Incumbent** — the best integer-feasible point found so far.
- **KKT conditions** — primal feasibility, dual feasibility, complementary slackness.
- **PDHG** — primal–dual hybrid gradient, the first-order saddle-point method.
- **Presolve / postsolve** — simplify the model before solving; map the answer back after.
- **Reduced cost** — $d_j = c_j - a_j^\top y$; the price change at which column $j$ enters the plan.
- **Restart** — resume an iterative method from a recent good point; turns sublinear into
  linear convergence on sharp problems.
- **Shadow price** — the dual value of a row: the objective's rate of change per unit of its
  right-hand side.
- **Warm start** — start a solve from a previous basis or point instead of from scratch.

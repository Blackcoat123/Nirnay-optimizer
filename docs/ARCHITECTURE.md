# NIRNAY — architecture

How the solver is put together, where each algorithm lives, and where the next engine plugs
in. PS26119 asks for a "transparent, extensible foundation"; this document is the map that
claim is checked against. The mathematics behind every method is in
[`THEORY.md`](THEORY.md), every citation in [`PROVENANCE.md`](PROVENANCE.md), every number in
[`BENCHMARKS.md`](BENCHMARKS.md) and [`RESULTS.md`](RESULTS.md), and the rules every change is
held to in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 1. The shape in one paragraph

A `Model` goes in, a `Solution` comes out, through one function:

```
solve(const Model&, const Options&) -> Solution        (src/core/solve.cpp)
```

`solve()` classifies the model by what it contains — integrality, a quadratic objective,
neither — and dispatches to an engine. Every engine consumes the same `Model`, produces the
same `Solution`, and is judged by the same code afterwards: `Solution::recompute_quality()`
re-measures feasibility, the reduced costs and the duality gap against the original model,
and the dispatcher's status guard downgrades any claim the measurement does not support.
That seam is what makes a new engine a bounded piece of work: it has to produce a
`Solution`, and it gets the same audit as the ones that exist.

```
          MPS / QPS / LP file              C API                 Python (ctypes)
                  │                          │                          │
                  └────────── readers (src/io) ─────────────────────────┘
                                         │
                                       Model ── classify: LP / MILP / QP / MIQP
                                         │
   LP ───── presolve (src/presolve) ─────┤
                                         ▼
      ┌──────────────── algorithm = auto | concurrent | dual-simplex | simplex | pdhg | ipm
      │  auto: dual simplex below 1,000 rows; the concurrent race at or above; pdhg with --gpu
      ▼
   ┌───────────────┐   ┌──────────────────────────────────────┐   ┌──────────────────────┐
   │ dual simplex  │   │ restarted PDHG (src/pdhg)            │   │ interior point       │
   │ primal simplex│   │   algorithm: pdhg.cpp (Halpern /     │   │ Mehrotra, sparse     │
   │ (src/simplex) │   │   adaptive), one driver              │   │ LDL^T (src/ipm, la)  │
   │  LU, devex,   │   │   backend: CPU (OpenMP)  or          │   └──────────┬───────────┘
   │  bound flips  │   │            CUDA (src/gpu, cuSPARSE)  │              │
   └───────┬───────┘   └──────────────────┬───────────────────┘              │
           │                              │ polish (IPM warm start)          │
           │                              ├──────────── crossover ◄──────────┘
           │                              │ (src/simplex/crossover.cpp)
           └───────── re-solve an unproven optimum from its own basis ───────┘
                                         │
                                     postsolve
                                         │
   MILP / MIQP ── branch and bound (src/mip): heap of open nodes, propagation, reliability
                  branching, warm-started dual node LPs (QP nodes for MIQP), root cuts,
                  rounding, diving, feasibility pump, RINS
   QP ─────────── convexity test (LDL^T), Condat–Vũ first-order QP (src/qp)
                                         │
                   Solution ── recompute_quality ── status guard ── certificates checked
                                         │
                    .sol / --stats JSON (src/io)        tools/verify_solution.py
```

## 2. Modules and their boundaries

| directory | what lives there | depends on | lines |
|---|---|---|---|
| `include/nirnay/` | the public headers: `model.hpp` (Model, Solution, BasisStatus), `options.hpp`, `tolerances.hpp` (every numerical constant, with its reason), `nirnay.h` (the C API), `solve_control.hpp` (progress and interrupt), `io.hpp`, `mip.hpp`, `qp.hpp`, `pdhg.hpp`, `ipm.hpp` | nothing | 2.1k |
| `src/core/` | `Model`/`Solution` implementation, `recompute_quality()`, the `solve()` dispatcher and its status guard, certificates, the **concurrent race** (`concurrent.cpp`), and the re-solve of an unproven optimum from its basis | everything below | 1.5k |
| `src/util/` | logging, the options registry (every option has a description, a default and a range, checked by a test), OpenMP thread control, the version stamp | – | 1.0k |
| `src/la/` | CSC/CSR sparse matrices with in-place scaling, the sparse LU (Markowitz threshold pivoting, product-form update, hyper-sparse FTRAN), the sparse LDLᵀ (approximate minimum degree), Ruiz + Pock–Chambolle equilibration | – | 2.4k |
| `src/io/` | MPS (fixed and free, RANGES, negative-UP convention, MARKER blocks, QUADOBJ, gzip), CPLEX-LP reader, `.sol` and `--stats` JSON writers | core | 2.3k |
| `src/presolve/` | reductions and the postsolve stack that reconstructs the primal AND the dual of the original model | core, la | 1.8k |
| `src/simplex/` | `simplex_core.hpp` (state shared by both loops: basis, factors, pricing weights, perturbation, warm start), the bounded primal and dual simplex, and **crossover** (`crossover.cpp`: basis identification from a point + dual-simplex cleanup) | core, la | 4.0k |
| `src/pdhg/` | the PDHG **driver** (`pdhg.cpp`: both schemes, restarts, primal weight, stopping, host re-verification), the **backend interface** (`backend.hpp`), the **CPU backend** (`cpu_backend.cpp`, OpenMP, deterministic reductions), and the residual definitions shared by host and device (`residuals.*`) | core, la | 1.6k |
| `src/gpu/` | the **CUDA backend** (`cuda_backend.cu`): device-resident vectors, cuSPARSE CSR products for Â and Âᵀ, fused kernels, deterministic two-pass reductions. Compiled only with `NIRNAY_ENABLE_CUDA` | pdhg | 0.9k |
| `src/ipm/` | Mehrotra predictor–corrector on the normal equations over the sparse LDLᵀ; warm start for the PDHG polish | core, la | 1.0k |
| `src/qp/` | convexity check (LDLᵀ of the Hessian, certificate on refusal), Condat–Vũ first-order convex QP | core, la | 0.5k |
| `src/mip/` | branch and bound (`branch_and_bound.cpp`), root cuts (`cuts.cpp`: GMI, lifted cover, integer rounding), primal heuristics (`heuristics.cpp`: feasibility pump, RINS) | core, simplex, qp | 3.0k |
| `src/api/` | the C API over `solve()`, with progress callbacks and interrupt; the Python bindings wrap this, not the C++ | core | 0.6k |
| `apps/nirnay-cli/` | `nirnay solve | info | options | version`, `--gpu`, `--stats`, `--write-sol`, `--progress-out`, `--option k=v`, Ctrl-C interrupt | api, io | 0.3k |
| `tools/` | `verify_solution.py`: re-parses the model with its own reader and checks the `.sol` file (bounds, rows, integrality, reduced costs, dual feasibility, complementary slackness, strong duality, Farkas vectors and rays). Shares no code with the solver, deliberately. `make_sbom.py`, provenance checks | – | 2.3k |
| `tests/` | 27 unit-test files; `oracles/` — a rational-arithmetic simplex and exact MILP branch and bound the float engines are fuzzed against; `robustness/` — the sweeps that find where the solver stops working | – | 14k |
| `bench/runners/` | fetchers (Netlib, MIPLIB, Mittelmann, Maros–Mészáros) and runners (Netlib, MIPLIB, Mittelmann, Maros–Mészáros, scale families, HiGHS comparison and cross-check, robustness), the analytic-optimum generators (random, staircase, refinery), the shared machine tag and commit identity (`machine.py`), and `make_benchmarks_doc.py` | – | 6.5k |

The dependency direction is strictly downward in that table: `la` knows nothing about
models, `simplex` knows nothing about integrality, `mip` knows nothing about file formats,
and the PDHG algorithm knows nothing about which hardware runs it. No file in `src/` reads
or links anything from another optimization solver.

## 3. The invariants everything rests on

**The frozen interface.** `Model`, `Solution` and `solve()` do not change without an explicit
note. Readers produce a `Model`; every engine consumes one and produces a `Solution`; the
writers, the verifier, the C API and the bindings consume a `Solution`. A new engine touches
`src/<engine>/` and one branch of the dispatcher.

**Nothing is reported that was not measured.** `recompute_quality()` recomputes the row
activities, the objective, the reduced costs and the KKT residuals against the ORIGINAL model
after postsolve, and the status guard turns an engine's "optimal" into "feasible" (or a
numerical error) when those numbers disagree with the claim. On the GPU, the stopping
decision is re-made on the host from a fresh unscaled measurement before the loop may stop.
Outside the process, `tools/verify_solution.py` repeats the audit with its own reader.

**Determinism.** Every single engine is deterministic at any thread count, and on the GPU on
a given device. `algorithm=concurrent` is the one documented exception (which engine wins
may depend on load; every answer is verified identically).

## 4. How an LP solve flows

1. **Read.** `src/io` produces a `Model` with column-major storage, bounds, integrality and an
   optional lower-triangular Hessian. Numbers are parsed with `std::from_chars` (correctly
   rounded, allocation-free), falling back to `strtod` for Fortran `D` exponents.
2. **Classify and choose.** `auto` is the dual simplex below `concurrent_min_rows` (1,000) rows
   and the concurrent race at or above it; `--gpu` with `auto` is PDHG on the device.
3. **Presolve.** Reductions recorded on a stack.
4. **Solve.**
   - *Simplex:* scaled internally (Ruiz, then Pock–Chambolle); the dual simplex starts from the
     slack basis or a warm start, boxes dual-infeasible columns, runs the bound-flipping ratio
     test with dual Devex pricing, perturbs costs on a degenerate stall, and hands over to the
     primal loop when it cannot finish honestly. Optimality only on fresh factors.
   - *PDHG:* scaled; `backend->spectral_norm()` sets the step; blocks of 64 Halpern iterations
     run in the backend without host decisions; every 64 iterations the backend measures the
     KKT sums in original units from the products it carries (no sparse product), the driver
     decides restart / stop, and a stop is confirmed on the host. Then the IPM polish when it
     stopped short, and crossover per the `crossover` option.
   - *Concurrent:* the three engines in threads, each with its own `SolveControl`; each racer's
     answer is tightened and judged by the status guard before the race sees it; the first
     judged optimum interrupts the rest.
   - *Any simplex optimum that measures short* is re-solved from its own basis with tighter
     internal tolerances and kept only if it then measures within tolerance.
5. **Postsolve.** Primal values in reverse record order; duals to a fixed point; reduced
   costs of touched columns recomputed as `c − Aᵀy`.
6. **Audit and report.** `recompute_quality()`, the status guard, the certificate check, then
   the `.sol` and JSON writers. The `.sol` file carries 17 significant digits.

For a MILP, step 4 is the tree: nodes chosen from a heap (depth-first while diving,
best-bound otherwise), propagation at each node, a root dive and — when it finds nothing —
the feasibility pump, RINS at geometrically spaced nodes, reliability branching, and every
child solved by the dual simplex from its parent's basis.

## 5. The GPU path in detail

```
 host                                            device (one CUDA stream)
 ────                                            ──────────────────────────
 scale model, build CSR of Â (and Â' = CSC of Â)
 make_gpu_backend(): upload Â, Â', bounds,  ───► CSR matrices, bounds, scaling, costs
   scaling, original data                        6 slots × (x, Â'y, y, Âx): iterate, output,
                                                 anchor, best, running sum, scratch
 spectral_norm()                            ───► power iteration (cuSPARSE SpMV × 2)
 loop:
   halpern_steps(64, k, τ, σ, ρ)            ───► 64 × [ primal kernel → SpMV(Â) →
                                                        dual+Halpern kernel → SpMV(Â') →
                                                        primal Halpern update ]
                                            ◄─── 3 scalars (fixed-point residual pieces)
   measure(output)                          ───► row + column KKT kernels, 2-pass reduction
                                            ◄─── 8 scalars
   restart_to / save_best                   ───► device-to-device copies
   if converged: download(output)           ◄─── x, y  → host re-measures on the ORIGINAL
                                                          model with fresh products
```

A device error at any point is recorded, every later backend call becomes a no-op, and the
solve ends as a numerical error with the CUDA message — never with stale numbers. A build
without CUDA, or a machine without a device, answers `--gpu` with one warning and the CPU
backend.

## 6. Where the next engines plug in

- **NLP / MINLP.** A `Model` today is linear constraints with an optional quadratic
  objective. A nonlinear engine would extend `Model` with constraint functions and gradients
  (an explicit interface change) and plug in at the same dispatcher seam; the MILP tree needs
  no change to search over it, since it only reads `col_value` and the bound.
- **A convex-QP interior point.** The sparse LDLᵀ and the Mehrotra loop exist; the QP form
  needs the augmented (quasi-definite) system in place of the normal equations.
- **GPU QP.** The PDHG backend interface is the natural place: a QP step adds a product with
  `Q` to the primal gradient.
- **Parallel tree search.** The concurrent race and the thread-safe `SolveControl` are the
  building blocks; the incumbent would be shared through the same kind of guarded channel.

## 7. Toolchain, as tested

- C++20, CMake ≥ 3.20, Ninja. GCC 13 on Ubuntu 24.04 (the DGX login and compute nodes);
  MSYS2 UCRT64 GCC on Windows through `scripts/configure.sh`, which refuses a compiler older
  than GCC 10.
- CUDA 12.8 (driver 570) for Blackwell `sm_100`, built by `scripts/build_cuda.sh`, which finds
  `nvcc` and builds in a clean environment; default architectures `80;90;100` (A100, H100,
  B200).
- Dependencies (all non-solver, table in `PROVENANCE.md`): fmt, CLI11, nlohmann/json,
  GoogleTest (tests only), zlib (optional), OpenMP (optional), CUDA runtime and cuSPARSE
  (optional). No BLAS, no LAPACK.
- `clang-format` 22.1.8 from pip, pinned.
- Python 3.10+ for the runners; `highspy` optional, for the comparison runs only.
- `scripts/reproduce.sh` takes a fresh clone to every claim; `bench/slurm/` holds the batch
  scripts for the B200 runs.

# NIRNAY

**An indigenous, GPU-accelerated mathematical optimization solver — LP, MILP, convex QP and
MIQP — written from mathematical foundations.**

Smart India Hackathon 2026 · Problem statement **SIH26119** — *Indigenous GPU-Accelerated
Optimization Solver (Sovereign Alternative to Xpress / CPLEX)* · Mangalore Refinery and
Petrochemicals Limited (MRPL).

*Nirṇaya* (निर्णय) means *decision*. An optimization solver is a decision engine: given what a
plant can do and what things cost, it decides what to do — and NIRNAY proves the decision
is right before it reports it.

> Not built on any existing solver. No source code from CPLEX, Gurobi, Xpress, HiGHS, SCIP,
> CBC/Clp, GLPK, OSQP, PDLP or cuPDLP was copied, linked or read; every algorithm is
> implemented from the published literature and cited beside its code. See
> [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the dependency table, the citation of every
> algorithm, the link line and the linked-library dump.

---

## Headline results

Every number below comes from a CSV in [`bench/results/`](bench/results/) that records the
instance's sha256, both objectives, the status, the time, the git commit and the machine.
The GPU rows were measured on **one 1g.45gb MIG slice of an NVIDIA B200 — about 1/7 of the
card** — on a DGX-B200 node (Intel Xeon Platinum 8570); a full card has several times the
bandwidth these runs had.

| What | Result | Evidence |
|---|---|---|
| **Hourly refinery-planning year** — 779,640 rows × 1,208,880 columns, 10 million nonzeros | **optimal in 4.0 s** of solver time on the GPU slice (13 s including reading a 421 MB file), relative error 1.6e-14 against the optimum known by construction. 32 CPU threads: 9.4 s. The previous engine reached 1.1e-6 in 134 s and proved nothing. | `scale-refinery-nirnay-gpu.csv`, `scale-refinery-nirnay-cpu32.csv` |
| **One million × one million LP** (random sparse, 5 million nonzeros) | **optimal in 233 s** on the GPU slice, relative error 5.6e-12. 32 CPU threads hit a 1,200 s limit without proving it. | `scale-random-nirnay-gpu.csv` |
| **Scale families** (random, staircase, refinery; 1,000 to 1,000,000 rows) | **every solve reaches the analytic optimum** — to a relative 5e-11 or better — on GPU and CPU alike | `scale-*-nirnay-*.csv` |
| **GPU vs 32 CPU threads**, same algorithm, same iterations | 20,000-row random: 17 s vs 145 s (8.4×); daily refinery year: 0.5 s vs 1.7 s | `scale-random-nirnay-*.csv`, `scale-refinery-nirnay-*.csv` |
| **Netlib, full 89-instance set** | **80 of 89** match the published optimum to 1e-6 **and** pass independent verification (was 78). Of the nine that differ from Netlib's table, eight agree with HiGHS to 2.4e-12 or better — the published table is the outlier — and the ninth, `pilot87`, does too since the fix described below. | `netlib-full-nirnay-dgx.csv` |
| **First-order engine efficiency** | the new PDHG needs 10–26× fewer iterations than the previous engine on the committed Netlib instances (e.g. `israel` 368,720 → 8,384) and reaches optimal on all nine with no finishing step | [`docs/THEORY.md`](docs/THEORY.md) §7.5 |
| **Branch and bound throughput** | 2.85× more nodes per second after replacing the scanned open list with a heap (profiled: bookkeeping was 76% of the solve) | [`docs/THEORY.md`](docs/THEORY.md) §10.1 |
| **MIPLIB 2017, 60 easy instances with a proven optimum**, 60 s each | 25 of 60 reach the published optimum, 14 of 60 also prove it — measured with the binary **before** the branch-and-bound heap and the new heuristics; the re-run with them is pending | `miplib-baseline-nirnay-dgx.csv` |
| **Convex QP (Maros–Mészáros, instances ≤ 200 rows)** | 25 of 49 match the reference and verify — the weakest area today, stated rather than hidden | `maros-meszaros-le200rows-nirnay.csv` |

A note on one Netlib instance, because it shows the method working: `pilot87` was reported as
`feasible` in the committed evidence. NIRNAY now re-solves an optimum whose measurement falls
short from its own final basis, which finds **301.710347333** — equal to HiGHS's
301.71034733310745 to 1e-15, accepted by the independent verifier — and shows that Netlib's
published 301.71072827 is off by 1.3e-6. (That fix landed after the Netlib CSV above was
written; the CSV still records `pilot87` as `feasible`.)

---

## What PS26119 asks for, and what NIRNAY provides

| Requirement | NIRNAY |
|---|---|
| LP, MILP, QP; modular for MIQP, NLP, MINLP | LP, MILP, convex QP **and MIQP** implemented; one `solve()` seam where a nonlinear engine plugs in ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §6) |
| Revised simplex | bounded **dual** simplex (default) and primal simplex: sparse Markowitz LU, hyper-sparse FTRAN, Devex, bound-flipping ratio test, perturbation, basis repair, iterative refinement |
| Interior-point methods | Mehrotra predictor–corrector over a from-scratch sparse LDLᵀ with approximate-minimum-degree ordering |
| GPU acceleration "where it provides measurable benefits" | **restarted PDHG on CUDA** — measured above; every vector stays on the device |
| Branch and bound, branch and cut, cutting planes | branch and bound with reliability branching and warm-started dual node LPs; root Gomory and lifted cover cuts |
| Presolve | eight reductions with dual-aware postsolve |
| Heuristics, advanced node selection | rounding, diving, **feasibility pump**, **RINS**; depth-first diving then best-bound from a heap |
| Sparse techniques, multi-core parallelism | sparse throughout; OpenMP (deterministic at any thread count); a **concurrent optimizer** racing engines in parallel |
| Numerical stability, reliable convergence | scaling, measured status guard, certificates for infeasible/unbounded, independent verification ([`docs/THEORY.md`](docs/THEORY.md) §12–13) |
| MIPLIB / Netlib / Mittelmann, compared with an established solver | runners and fetchers for all three plus Maros–Mészáros; HiGHS as a separate-process comparison |
| API or CLI | a CLI, a C API, and Python bindings |
| Refinery, blending, planning, dispatch, supply chain | case studies and a generator for refinery-planning LPs of any horizon with the optimum known exactly |

---

## The engines

**LP.**
- **Dual simplex** (the default below 1,000 rows) and **primal simplex**. Exact vertices, a basis, and shadow prices.
- **Interior point** (`algorithm=ipm`).
- **Restarted PDHG** (`algorithm=pdhg`, `--gpu` for the CUDA backend). A first-order method that scales to millions of variables. Its default scheme is reflected restarted Halpern PDHG with a constant step; PDLP's adaptive scheme is also available.
- **Concurrent** (`algorithm=concurrent`, and what `auto` does at or above 1,000 rows). Races the dual simplex, PDHG (on the GPU when present) and the interior point in parallel threads. The first answer proved optimal wins and interrupts the rest.
- **Finishing steps.** PDHG answers that stop short are polished by the interior point. First-order answers are **crossed over** to an optimal basis (`crossover=auto|on|off`). A simplex optimum that measures short is re-solved from its own basis with tighter tolerances.

**MILP.** Branch and bound with these components:
- bound propagation and reliability branching (pseudocosts plus strong branching);
- warm-started dual simplex node LPs;
- rounding, diving, the feasibility pump and RINS;
- optional root cuts;
- gap targets.

**QP and MIQP.** Convexity is decided first. A non-convex model is refused with a certificate, never solved to a local point. Convex models go to a Condat–Vũ primal–dual method, and MIQP to branch and bound over QP relaxations with conservative pruning.

**Every answer** is re-measured against the original model before it's reported:
- `optimal` carries a dual certificate;
- `infeasible` carries a Farkas vector;
- `unbounded` carries a ray.

All three can be checked with `tools/verify_solution.py`, which shares no code with the solver.

The mathematics of all of this is in [`docs/THEORY.md`](docs/THEORY.md).

---

## Quick start

### Build (CPU)

Requires CMake ≥ 3.20, Ninja and a C++20 compiler (GCC ≥ 10, Clang ≥ 12, MSVC ≥ 19.30).

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build -j        # 483 tests
```

### Build (GPU)

Requires the CUDA toolkit, version 12.8 or newer for a B200 (`sm_100`), no newer than your driver supports (`nvidia-smi` shows the limit). Compiling needs no GPU.

```bash
scripts/build_cuda.sh                 # build-cuda/, architectures 80;90;100 (A100, H100, B200)
scripts/build_cuda.sh build-b200 100  # a B200-only build
```

`build_cuda.sh` finds `nvcc` on `PATH`, under `$CUDA_HOME`, under `/usr/local/cuda*`, or in a conda environment. On a cluster that has the driver but no toolkit, create the environment:

```bash
conda create -n nirnay -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-cccl \
                       libcusparse-dev cmake ninja
```

Then run the script without activating the environment. The script explains why at its top.

### Solve

```bash
./build/nirnay solve demo/crude_blend.mps --write-sol blend.sol --stats blend.json
./build/nirnay solve model.mps --gpu                          # PDHG on the GPU
./build/nirnay solve model.mps --option algorithm=concurrent  # race the engines
./build/nirnay solve model.mps --progress-out progress.jsonl  # tail -f it during a long solve
./build/nirnay options                                        # every option, with its default
python3 tools/verify_solution.py demo/crude_blend.mps blend.sol   # independent check
```

`solve` exits `0` for optimal, `1` for a limit or a proven infeasible/unbounded model, `3` when the file can't be read, and `5` for a numerical or model error. Ctrl-C stops a solve and returns the best point found so far.

The solution file carries the **shadow price of every row**. On the blending model, those are the numbers a refinery planner acts on: what one more unit of diesel commitment costs, and what the sulphur specification is worth.

### Python

```python
import nirnay                                   # bindings/python, ctypes over the C API
model = nirnay.Model.read("data/netlib/afiro.mps")
result = model.solve(algorithm="concurrent")
print(result.status, result.objective)          # optimal -464.753142857...
```

---

## Running on the B200 cluster (Slurm)

The DGX-B200 is shared, so batch jobs are the way in. Build on the login node, then:

```bash
sbatch bench/slurm/b200_evidence.sbatch        # a full B200: scale, Netlib, Mittelmann,
                                               # MIPLIB, Maros-Meszaros
sbatch -p small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 \
       --export=ALL,NIRNAY_MACHINE_TAG=dgx-b200-mig1g45 bench/slurm/b200_evidence.sbatch
```

`NIRNAY_MACHINE_TAG` is recorded in every CSV row. Name a MIG slice as a slice.

---

## Benchmarks: fetch, run, reproduce

```bash
python3 bench/runners/fetch_data.py --set full          # Netlib, 89 instances + published optima
python3 bench/runners/fetch_miplib.py --count 60        # MIPLIB 2017 easy instances with =opt=
python3 bench/runners/fetch_mittelmann.py --set full    # Mittelmann LP set (~26 GB decoded)
python3 bench/runners/fetch_maros_meszaros.py           # 138 convex QPs + reference optima

python3 bench/runners/netlib.py --time-limit 120
python3 bench/runners/miplib.py --time-limit 60
python3 bench/runners/mittelmann.py --time-limit 300
python3 bench/runners/maros_meszaros.py --time-limit 60
python3 bench/runners/scale.py --structure refinery --sizes 12 365 8760 --engines pdhg \
        --solver-option gpu=true                        # optimum known by construction
python3 bench/runners/cross_check_highs.py              # every disagreement vs HiGHS
scripts/reproduce.sh                                    # a fresh clone to every claim, offline
```

Each runner writes a CSV to `bench/results/` with the instance hash, both objectives, the gap, the status, the time, the iterations or nodes, the commit (suffixed `-dirty` when the tree is modified) and the machine. `docs/BENCHMARKS.md` is generated from those CSVs. The nine Netlib instances the demo uses are committed, so the demo needs no network.

```bash
demo/run_sih_demo.sh        # the PS26119 walkthrough, in the problem statement's order
demo/run_demo.sh share2b    # solve one instance live, then verify it independently
```

---

## Useful options

| Option | Default | What it does |
|---|---|---|
| `algorithm` | `auto` | `auto`, `dual-simplex`, `simplex`, `ipm`, `pdhg` or `concurrent`. `auto` runs the dual simplex below `concurrent_min_rows` (1,000) and the race above; with `--gpu` it runs PDHG |
| `gpu` / `--gpu` | off | PDHG on the CUDA device (`gpu_device` picks one) |
| `pdhg_method` | `auto` (= `halpern`) | `halpern` (reflected restarted Halpern, constant step) or `adaptive` (PDLP) |
| `crossover` | `auto` | `on`, `off` or `auto`. `auto` runs on small models, and on any answer not already optimal, within a bounded budget |
| `time_limit`, `iteration_limit`, `node_limit` | none | limits; a limit returns the best point as exactly that |
| `mip_relative_gap` | 1e-4 | the MIP stops as optimal within this gap |
| `mip_heuristics` | on | feasibility pump and RINS |
| `enable_root_cuts` | off | Gomory and cover cuts at the root (off by measurement) |
| `threads` | 1 | OpenMP threads (0 = all cores); answers identical at any count |
| `presolve` | on | reductions with dual-aware postsolve |

`nirnay options` prints every option with its default, its range and the measurement behind it.

---

## Repository layout

```
include/nirnay/   public headers: Model, Solution, Options, tolerances, nirnay.h (the C API)
src/core          solve() dispatcher, status guard, certificates, the concurrent race
src/simplex       dual and primal simplex, crossover
src/pdhg          PDHG driver (Halpern / adaptive), backend interface, CPU backend
src/gpu           CUDA backend (cuSPARSE, fused kernels, deterministic reductions)
src/ipm           Mehrotra interior point
src/la            sparse matrices, LU, LDL^T with AMD, equilibration
src/mip           branch and bound, root cuts, feasibility pump, RINS
src/qp            convexity test, Condat-Vu convex QP
src/presolve      reductions and postsolve
src/io            MPS / QPS / LP readers, .sol and JSON writers
src/api           C API;  bindings/python: ctypes bindings;  apps/nirnay-cli: the CLI
tools/            verify_solution.py - the independent verifier
tests/            unit tests, the exact rational oracle, robustness sweeps
bench/            fetchers, runners, generators, results (CSVs), Slurm scripts
demo/             the PS26119 walkthrough and small industrial models
data/             committed Netlib instances, case studies, reference manifests
docs/             theory, architecture, provenance, benchmarks, coverage, contributing
```

## Documentation

| Document | What it is |
|---|---|
| [`docs/THEORY.md`](docs/THEORY.md) | the mathematics: duality, simplex, interior point, PDHG and its GPU mapping, crossover, concurrency, branch and bound, heuristics, QP, robustness, verification |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the map of the code, the solve flow, the GPU data path, where new engines plug in |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | benchmark tables generated from the CSVs |
| [`docs/PS26119_COVERAGE.md`](docs/PS26119_COVERAGE.md) | the problem statement, requirement by requirement |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | dependencies, algorithm citations, link line, SBOM, judgement calls |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | the rules every change is held to: provenance, evidence, numerics, determinism |

---

## What NIRNAY does not do yet

Stated plainly, because a solver that is vague about its limits isn't one an industrial user can plan around.

| | |
|---|---|
| **Convex QP accuracy** | The QP engine is first-order, and on the small Maros–Mészáros instances 25 of 49 reach the reference. An interior point for QP (on the augmented system) is the next engine. |
| **Hard MILPs** | Branch and bound finds good incumbents far more often than it proves optimality on the harder MIPLIB instances. There are no MIR cuts, no cuts below the root, and no parallel tree search yet. |
| **Large LPs with the simplex** | The dual simplex's iteration rate falls off beyond ~10,000 rows, so crossover to a vertex is cheap on small and medium models but not at a million rows. At that size the answer is PDHG's, verified to 1e-7 absolute feasibility and a 1e-9 relative gap, without a basis. |
| **Mittelmann's large LPs** | The runner exists and the set is fetched. A full-card B200 run with the concurrent engine is pending; the last committed laptop run finished 0 of 8. |
| **NLP / MINLP** | Not attempted. The architecture leaves the seam for it. |
| **Determinism of `concurrent`** | Every single engine is bit-for-bit deterministic. The race is not: on a model with several optimal vertices, which one is reported can vary between runs, though each is verified. |

## Licence

Apache-2.0.

# Contributing to NIRNAY — the engineering rules

These are the rules every change to NIRNAY is held to. They exist because this is numerical
code: a wrong answer compiles, runs, prints, and looks completely correct. There is no stack
trace. The rules below are how a wrong answer gets caught before it reaches a user — and how
a judge, a reviewer or an industrial adopter can check that it was.

---

## 1. Provenance — the one hard rule

PS26119 requires a solver "built from scratch from mathematical foundations", not on top of
an existing solver. So:

**No source code from an optimization solver may be copied, vendored, linked, or read.**
That covers CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP, cuPDLP / cuPDLP-C /
cuPDLPx, OR-Tools GLOP and CP-SAT, cuOpt, Gurobi, CPLEX and Xpress — not their simplex, not
their cuts, not even their MPS reader.

Explicitly allowed, and used:

1. Knowledge of the algorithms, from papers, theses, textbooks, lecture notes and solver
   *user manuals*. Every algorithm carries its citation in a comment above its
   implementation, and in the algorithm table of [`PROVENANCE.md`](PROVENANCE.md).
2. Non-solver libraries, licence permitting: fmt, CLI11, nlohmann/json, GoogleTest, zlib,
   and NVIDIA's vendor libraries (CUDA runtime, cuSPARSE) — sparse BLAS, not solvers.
3. Imitating an API's *shape* (create / set / solve / query, string options) — interface
   compatibility, not derivation.
4. Benchmark instances and published reference optima: Netlib, MIPLIB, Mittelmann,
   Maros–Mészáros, QPLIB.
5. Running an established solver **as a separate process** to compare answers against
   (`bench/runners/cross_check_highs.py`, `compare.py`). Its code is never linked or read.

Anything unclear is logged under "Judgement calls" in `PROVENANCE.md` with its reasoning —
never decided silently.

## 2. Evidence — nothing is claimed that was not measured

- A benchmark number, pass rate or speed-up is only quoted if a command produced it, and
  every such command writes a CSV to `bench/results/` recording: the instance, its sha256,
  our objective, the reference objective, the absolute and relative gap, the status, wall
  time, iterations or nodes, the git commit (suffixed `-dirty` when the tree is modified)
  and a machine tag. No CSV, no claim.
- `docs/BENCHMARKS.md` is generated from those CSVs by `bench/runners/make_benchmarks_doc.py`
  so that it cannot drift from them.
- A failing test is reported as failing. A tolerance is never loosened to make a test pass
  without saying so and giving the numerical justification.
- Every reported status is checked against a *measurement* of the point it comes with
  (`src/core/status_guard.hpp`), and every written solution can be re-checked by
  `tools/verify_solution.py`, which shares no code with the solver.
- Assume output is wrong until a benchmark, the exact rational oracle, or an independent
  solver says otherwise.

## 3. Definition of done for any change

- [ ] `cmake --build build -j` is clean under `-Wall -Wextra -Werror` (and the rest of the
      warning set in `CMakeLists.txt`)
- [ ] `ctest --test-dir build -j` passes — including in parallel
- [ ] a CUDA build (`scripts/build_cuda.sh`) compiles, and its GPU tests pass on a GPU
- [ ] ASan/UBSan clean on the touched path (`-DNIRNAY_SANITIZE=ON`)
- [ ] the Netlib pass count did not drop; the rational-oracle fuzz still reports 0
      mismatches
- [ ] the algorithm is cited in a code comment
- [ ] `docs/PROVENANCE.md` is updated if a dependency or an algorithm was added
- [ ] a conventional commit, one logical change

## 4. Numerical conventions

No magic numbers: every tolerance lives in `include/nirnay/tolerances.hpp`, with its
reason. Primal feasibility 1e-7 · dual feasibility 1e-7 · integrality 1e-6 · MIP relative
gap 1e-4, absolute 1e-6 · pivot / zero drop 1e-11 · Markowitz threshold 0.01. Double
precision everywhere — no `float` in the numerical core, on the CPU or on the GPU.

Feasibility is judged on the *scaled* violation (each violation divided by the magnitude of
the terms it was computed from), and the absolute one is what gets printed — see
`Solution::primal_infeasibility_scaled` in `include/nirnay/model.hpp` for why.

## 5. Frozen interfaces

- `nirnay::Model` — what the readers produce and every engine consumes
- `nirnay::Solution` — what every engine produces
- `solve(const Model&, const Options&) -> Solution` — the one entry point, and the seam
  where new engines plug in

These change only with an explicit note in the change that says so (see the
`crossover_iterations` field for how an addition is made). `tools/verify_solution.py` reads
the written `.sol` file only; it never links the C++.

## 6. Determinism

Every single engine is deterministic: the same model and options give the same answer, bit
for bit, at any thread count (OpenMP loops write each output from one thread; reductions sum
fixed chunks in a fixed order), and on the GPU on a given device. The one deliberate
exception is `algorithm=concurrent`, where whichever engine proves optimality first wins —
every answer is verified the same way, but on a model with several optimal vertices the
vertex reported may differ between runs.

## 7. Workflow

Branch per task, pull request into `main`, squash merge. No source file over ~600 lines
where it can be helped.

Formatting is gated with clang-format **22.1.8** from pip (its output changes between major
versions, so a distro clang-format "fixes" the tree into a state the gate rejects):

    scripts/format.sh            # rewrite in place
    scripts/format.sh --check    # what the gate runs

The CPU build must work with zero CUDA installed: all GPU code is behind
`NIRNAY_ENABLE_CUDA`, and `--gpu` on a build or machine without a device warns once and
runs on the CPU.

## 8. Toolchains

`scripts/configure.sh` picks a C++20 compiler (GCC ≥ 10, Clang ≥ 12, MSVC ≥ 19.30) rather
than trusting `PATH` order, which matters on Windows boxes carrying an old MinGW. For the GPU
build, `scripts/build_cuda.sh` finds `nvcc` (system CUDA, `$CUDA_HOME`, or a conda
environment) and builds in a clean environment — see the comments at its top for why a
conda compiler environment must not be *activated* while linking.

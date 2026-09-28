// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the seam between the PDHG algorithm and the hardware it runs on.
//
// WHY THERE IS A SEAM. Restarted PDHG spends essentially all of its time in two sparse
// matrix-vector products and a handful of element-wise passes per iteration. None of that
// has a serial dependency inside an iteration, which is what makes it the engine that can
// go on a GPU at all (the simplex cannot: every pivot depends on the one before it). The
// decisions BETWEEN iterations - accept a step, restart, stop - are scalar logic that
// belongs on the host.
//
// So the algorithm lives once, in pdhg.cpp, and talks to a Backend that owns the vectors
// and does the arithmetic. The CPU backend (cpu_backend.cpp) runs the passes under OpenMP;
// the CUDA backend (src/gpu/cuda_backend.cu) keeps every vector resident on the device and
// hands back only scalars. Both implement the same operations with the same formulas, so a
// run on either is the same algorithm, and test_pdhg_backends.cpp pins them to each other.
//
// WHAT CROSSES THE SEAM. Scalars, every iteration at most (three numbers for an adaptive
// step's acceptance test), and whole vectors only when the driver asks for a point to
// report or to verify on the host. A GPU run on a million-row model therefore moves a few
// bytes per iteration across PCIe, not megabytes.
//
// THE ARITHMETIC. The problem is the scaled one, Ahat = Dr A Dc (la/scaling.hpp):
//
//     min_{xhat in Xhat} max_{yhat}   chat' xhat + yhat' Ahat xhat - sigma_Chat(yhat)
//
// and every iterate is carried together with its matrix products, Ahat xhat and
// Ahat' yhat, because the products are linear in the iterate. That is what lets an
// iteration cost two sparse products instead of three, lets the Halpern scheme update its
// products without a third, and lets a measurement in ORIGINAL units - which is the only
// kind this project reports - run with no sparse product at all:
//
//     x = Dc xhat,  y = Dr yhat,  A x = Dr^-1 (Ahat xhat),  A' y = Dc^-1 (Ahat' yhat).
//
// Written from the papers cited in pdhg.cpp; no solver's source was consulted.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nirnay/types.hpp"

namespace nirnay::pdhg {

/// A compressed sparse matrix by its outer dimension: CSR when outer is the row count.
/// Views only - the driver owns the storage and keeps it alive for the backend's lifetime.
struct CompressedView {
  Index outer = 0;
  Index inner = 0;
  const Index* starts = nullptr;   ///< outer + 1 entries
  const Index* indices = nullptr;  ///< starts[outer] entries
  const double* values = nullptr;  ///< starts[outer] entries
};

/// Everything a backend needs, as views into the driver's arrays.
struct ScaledProblem {
  Index rows = 0;
  Index cols = 0;

  CompressedView by_rows;     ///< Ahat, rows x cols, row-compressed
  CompressedView by_columns;  ///< Ahat', cols x rows, row-compressed (= Ahat's CSC)

  // The scaled problem the iteration runs on, in minimise space.
  const double* cost = nullptr;       ///< chat, cols
  const double* col_lower = nullptr;  ///< cols
  const double* col_upper = nullptr;  ///< cols
  const double* row_lower = nullptr;  ///< rows
  const double* row_upper = nullptr;  ///< rows

  // The map back to original units, and the original data a measurement is taken against.
  const double* col_scale = nullptr;  ///< Dc, cols
  const double* row_scale = nullptr;  ///< Dr, rows
  const double* original_cost = nullptr;       ///< minimise-space c, cols
  const double* original_col_lower = nullptr;  ///< cols
  const double* original_col_upper = nullptr;  ///< cols
  const double* original_row_lower = nullptr;  ///< rows
  const double* original_row_upper = nullptr;  ///< rows
};

/// Which point a backend operation is about.
enum class Point : std::uint8_t {
  kIterate,  ///< the current iterate (for Halpern, the anchored iterate z_k)
  kOutput,   ///< the last PDHG output T(z): a trial step (adaptive) or T(z_k) (Halpern)
  kAverage,  ///< the running average since the last restart (adaptive only)
  kBest,     ///< the best point saved so far
};

/// Squared movements and their interaction, all in scaled units. For an adaptive trial
/// step they are (x'-x), (y'-y) and (y'-y)' Ahat (x'-x); for a Halpern step the same with
/// z - T(z); for a distance to the anchor only the two squares are filled.
struct StepMeasure {
  double dx_squared = 0.0;
  double dy_squared = 0.0;
  double cross = 0.0;
};

/// The raw sums a KKT measurement in ORIGINAL units is made of. pdhg.cpp turns them into
/// relative residuals with exactly the normalisations its host-side evaluate() uses, so a
/// point measured on the device and re-measured on the host lands on the same verdict up to
/// rounding - and the host measurement is the one that decides what is reported.
struct KktSums {
  double primal_violation_squared = 0.0;  ///< sum over rows of (bound violation of A x)^2
  double dual_violation_squared = 0.0;    ///< reduced costs and row prices no bound absorbs
  double bound_contribution = 0.0;        ///< sum_j d_j * (l_j or u_j)
  double support = 0.0;                   ///< sigma_C(y)
  double primal_objective = 0.0;          ///< c' x
  double complementarity = 0.0;           ///< max |multiplier| * slack (NaN products skipped)
};

class Backend {
 public:
  virtual ~Backend() = default;

  /// "pdhg-cpu" or "pdhg-gpu": what a Solution's `algorithm` field reports.
  [[nodiscard]] virtual std::string name() const = 0;
  /// One line for the log: device, threads, memory.
  [[nodiscard]] virtual std::string description() const = 0;

  /// ||Ahat||_2 by power iteration on Ahat' Ahat, from a seeded random start, rounded up
  /// by 1% because an underestimate is what makes PDHG diverge.
  [[nodiscard]] virtual double spectral_norm(int iterations, unsigned seed) = 0;

  /// Set the iterate, the anchor and the best point to (x, y), scaled, and compute their
  /// products. Clears the running average.
  virtual void initialize(const std::vector<double>& x, const std::vector<double>& y) = 0;

  // ---- The adaptive-step scheme ([PDLP] section 3.1) --------------------------------------

  /// One PDHG step from the iterate into the output slot, with its movement measured.
  virtual StepMeasure trial_step(double tau, double sigma) = 0;
  /// The output becomes the iterate; its A'y is computed; the running average absorbs it.
  virtual void accept_trial() = 0;

  // ---- The reflected Halpern scheme (Lu & Yang 2024) ----------------------------------------

  /// `count` steps  z <- w ((1 + rho) T(z) - rho z) + (1 - w) z0,  w = (k+1)/(k+2),  with k
  /// running from `first_k`. The last step also measures z - T(z) for fixed_point_measure().
  virtual void halpern_steps(Count count, Count first_k, double tau, double sigma,
                             double reflection) = 0;
  /// z - T(z) at the last Halpern step taken.
  [[nodiscard]] virtual StepMeasure fixed_point_measure() const = 0;

  // ---- Measurement, restarts and reporting --------------------------------------------------

  /// KKT sums of a point, in ORIGINAL units, with no sparse product.
  [[nodiscard]] virtual KktSums measure(Point point) = 0;
  /// ||x_p - x0||^2 and ||y_p - y0||^2 against the anchor (the last restart point).
  [[nodiscard]] virtual StepMeasure distance_to_anchor(Point point) = 0;
  /// Make `point` the iterate and the anchor; clear the running average.
  virtual void restart_to(Point point) = 0;
  /// Remember `point` as the best so far.
  virtual void save_best(Point point) = 0;
  /// Copy `point` to the host, scaled.
  virtual void download(Point point, std::vector<double>* x, std::vector<double>* y) = 0;
  /// Iterates in the running average.
  [[nodiscard]] virtual Count averaged() const = 0;

  /// Empty while the backend is healthy; otherwise the first device error it met, after
  /// which its operations do nothing. The driver checks this at every measurement and ends
  /// the solve as a numerical error with this text, rather than reporting stale numbers.
  [[nodiscard]] virtual std::string failure() const { return {}; }
};

/// The OpenMP backend. Every parallel loop writes each output element from one thread, and
/// every reduction sums fixed-size chunks in a fixed order, so the answer is bit-identical at
/// any thread count.
[[nodiscard]] std::unique_ptr<Backend> make_cpu_backend(const ScaledProblem& problem);

/// The CUDA backend, when this build has one and a device answers. Returns null and says why
/// in `why` otherwise; the caller falls back to the CPU and logs the reason.
[[nodiscard]] std::unique_ptr<Backend> make_gpu_backend(const ScaledProblem& problem,
                                                        int device, std::string* why);

}  // namespace nirnay::pdhg

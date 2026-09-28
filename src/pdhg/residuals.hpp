// SPDX-License-Identifier: Apache-2.0
// NIRNAY - how far a PDHG point is from optimal, measured in the ORIGINAL problem.
//
// Two measurements share these definitions. The backend's (backend.hpp) is taken every few
// dozen iterations, from the scaled products it already holds, and decides when to restart
// and when to look closer. The host's, evaluate() below, re-derives everything from the
// unscaled model with fresh sparse products; it runs on a point the backend says has
// converged, and on the point finally reported, and it is the only measurement a status is
// ever set from. A device and a host can differ in the last bits of a sum; the verdict is
// the host's.
#pragma once

#include <algorithm>
#include <vector>

#include "nirnay/model.hpp"
#include "nirnay/tolerances.hpp"

#include "pdhg/backend.hpp"

namespace nirnay::pdhg {

/// Convergence measured in the ORIGINAL problem, never the scaled one. Reporting residuals
/// from the scaled problem would let a well-chosen preconditioner flatter the result.
struct Residuals {
  double primal = 0.0;  ///< relative primal infeasibility
  double dual = 0.0;    ///< relative dual infeasibility
  double gap = 0.0;     ///< relative primal-dual objective gap
  double primal_objective = 0.0;
  double dual_objective = 0.0;

  /// The same violations UNSCALED. A relative residual divides by (1 + ||bounds||), so on a
  /// model whose right-hand sides run to 1e4 a relative 1e-8 still permits an absolute
  /// violation around 1e-4. That is standard and fine as a stopping rule - it is what the
  /// PDLP literature uses - but it is NOT the standard the rest of this project reports
  /// against, and conflating the two is how the solver ends up stamping "optimal" on a point
  /// tools/verify_solution.py then rejects.
  double absolute_primal = 0.0;
  double absolute_dual = 0.0;

  /// The duality gap normalised the way tools/verify_solution.py normalises it, by
  /// max(1, |primal objective|), rather than the PDLP convention of 1 + |primal| + |dual|.
  /// The two differ by roughly a factor of two, which is more than enough for this engine to
  /// pass its own optimality test and fail the verifier's on the same point. The PDLP form
  /// stays as the stopping rule because that is the literature convention; the claim is
  /// judged by the verifier's form because that is what will be checked.
  double gap_as_verified = 0.0;

  /// max |multiplier| * slack over rows and columns - the same product form
  /// tools/verify_solution.py uses. The duality gap implies this only in the limit, so a
  /// point can show a tiny gap and still price a constraint it is not sitting on.
  double complementarity = 0.0;

  [[nodiscard]] double worst() const { return std::max({primal, dual, gap}); }

  /// Has the run met the tolerance the CALLER asked for, AND is the point actually feasible
  /// in absolute terms? Both are required to stop.
  ///
  /// The absolute half is not pedantry. kFeasible in nirnay::Solution asserts that a
  /// feasible point is being reported, so stopping on a relative residual alone would let
  /// this engine claim feasibility for a point that misses the project's own primal
  /// tolerance - a weaker claim than kOptimal, but still one the verifier rejects.
  [[nodiscard]] bool meets_request(double tolerance) const {
    return primal <= tolerance && dual <= tolerance && gap <= tolerance &&
           absolute_primal <= tol::kPrimalFeasibility;
  }

  /// Would this point survive independent verification? These are the project's own
  /// tolerances from include/nirnay/tolerances.hpp, the same ones the .sol file is judged
  /// against, and meeting them is the ONLY basis on which this engine claims kOptimal.
  [[nodiscard]] bool meets_project_standard() const {
    return absolute_primal <= tol::kPrimalFeasibility &&
           absolute_dual <= tol::kDualFeasibility && gap_as_verified <= tol::kDualityGap &&
           complementarity <= 1e-6;
  }
};

/// The unscaled problem, held once so the convergence test does not rebuild it.
struct Problem {
  const Model* model = nullptr;
  std::vector<double> cost;  ///< minimise-space objective
  double bound_norm = 0.0;   ///< ||finite row bounds||, for the relative primal residual
  double cost_norm = 0.0;    ///< ||c||, for the relative dual residual
};

/// Build the unscaled problem for `model` (costs folded into minimise space).
[[nodiscard]] Problem make_problem(const Model& model);

/// Measure (x, y), in original units, against the unscaled model with fresh sparse products.
/// `activity` and `reduced` receive A x and c + A' y. This is the host measurement every
/// reported status is set from.
[[nodiscard]] Residuals evaluate(const Problem& problem, const std::vector<double>& x,
                                 const std::vector<double>& y, std::vector<double>* activity,
                                 std::vector<double>* reduced);

/// The same quantities from a backend's sums, normalised exactly as evaluate() does.
[[nodiscard]] Residuals from_sums(const Problem& problem, const KktSums& sums);

}  // namespace nirnay::pdhg

// SPDX-License-Identifier: Apache-2.0
// NIRNAY - crossover: from a first-order or interior-point answer to an optimal basis.
//
// References:
//   Megiddo, "On finding primal- and dual-optimal bases", ORSA J. Computing 3(1), 1991 -
//     that an optimal basis can be recovered from an optimal primal-dual pair in strongly
//     polynomial time, which is what makes crossover a finishing step rather than a solve.
//   Bixby & Saltzman, "Recovering an optimal LP basis from an interior point solution",
//     Operations Research Letters 15(4), 1994 - basis identification from the point, then a
//     simplex cleanup from the identified basis.
//   El-Bakry, Tapia & Zhang, "A study of indicators for identifying zero variables in
//     interior-point methods", SIAM Review 36(1), 1994 - ranking variables by how far their
//     primal value sits from its bound against how large their reduced cost is.
//
// WHY. PDHG and the interior point converge to a TOLERANCE: their answer is a point near the
// optimal face, with duals near the optimal ones, and no basis. A refinery planner wants a
// vertex - an operating plan with exact shadow prices, the one the simplex would report -
// and branch and bound wants a basis to warm-start from. Crossover turns the first answer
// into the second: identify the basis the point is sitting on, then let the dual simplex
// finish from there. When the point is good the simplex needs a few pivots, not a solve.
//
// THE CONTRACT. Crossover never makes an answer worse. It replaces the first-order answer
// only with a simplex answer that is OPTIMAL - fresh factors, measured by the same status
// guard as every other engine - and otherwise leaves it untouched with the reason appended.
#pragma once

#include "nirnay/logging.hpp"
#include "nirnay/model.hpp"
#include "nirnay/options.hpp"
#include "nirnay/solve_control.hpp"
#include "nirnay/timer.hpp"

#include "simplex/primal_simplex.hpp"

namespace nirnay {

/// The basis a primal-dual point is sitting on: exactly num_rows basic variables, the
/// structurals and row logicals whose primal distance to their nearest bound is largest
/// against the magnitude of their reduced cost (for a row, its price). Every other variable
/// is placed at the bound it is nearest. `point` needs col_value, col_dual and row_dual in
/// the model's own units and sign convention; row_activity is recomputed if absent.
[[nodiscard]] WarmStart identify_basis(const Model& model, const Solution& point);

/// Whether the `crossover` option (on / off / auto) asks for a crossover of this answer.
[[nodiscard]] bool crossover_wanted(const Solution& approximate, const Model& model,
                                    const Options& options);

/// Finish `approximate` with the dual simplex from identify_basis(). On success the answer
/// becomes the simplex's optimal vertex, with a basis, `algorithm` suffixed "+crossover" and
/// the pivots recorded in crossover_iterations; otherwise it is left as it was and the
/// message says why. Honours `crossover_max_seconds` and whatever `time_limit` has left.
void crossover(Solution* approximate, const Model& model, const Options& options,
               Logger& logger, const Timer& timer, SolveControl* control);

}  // namespace nirnay

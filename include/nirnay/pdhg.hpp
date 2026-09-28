// SPDX-License-Identifier: Apache-2.0
// NIRNAY - restarted PDHG, the first-order LP engine.
#pragma once

#include "nirnay/logging.hpp"
#include "nirnay/model.hpp"
#include "nirnay/options.hpp"
#include "nirnay/solve_control.hpp"

namespace nirnay::pdhg {

/// Solve an LP with restarted primal-dual hybrid gradient.
///
/// Requires a model with no integrality and no quadratic objective; the solve() dispatcher
/// checks that. Never throws: every failure comes back as a status.
///
/// `control` (#223): optional progress/interrupt channel, polled at the same point the
/// iteration loop already checks its time limit.
[[nodiscard]] Solution solve_pdhg(const Model& model, const Options& options, Logger& logger,
                                  SolveControl* control = nullptr);

}  // namespace nirnay::pdhg

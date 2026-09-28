// SPDX-License-Identifier: Apache-2.0
#include "nirnay/solve_control.hpp"

namespace nirnay {

const char* to_string(SolvePhase phase) noexcept {
  switch (phase) {
    case SolvePhase::kPresolve: return "presolve";
    case SolvePhase::kLp: return "lp";
    case SolvePhase::kTree: return "tree";
  }
  return "unknown";
}

}  // namespace nirnay

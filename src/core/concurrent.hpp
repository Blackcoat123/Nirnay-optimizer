// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the concurrent LP optimizer: every engine at once, the first proof wins.
//
// WHY. No LP engine is best everywhere, and this project has measured exactly where each
// one wins (docs/BENCHMARKS.md): the dual simplex takes a Netlib instance in milliseconds
// and cannot finish a 32,000-row refinery year in five minutes; PDHG on a GPU solves that
// year in seconds and is slower than the simplex on anything small; the interior point sits
// between them on structured models. Choosing by a size threshold guesses; racing measures.
// Commercial solvers ship the same idea as their default for large LPs.
//
// HOW. Each engine runs in its own thread against the same (presolved) model, with its own
// SolveControl and a silent logger. The first to return OPTIMAL - with the status guard of
// every engine behind it, so optimal means the same thing whoever wins - interrupts the
// others through their controls, which reach inside the simplex's LU and the interior
// point's ordering and factorization (#197, #208, #223). If none proves optimality, the
// best answer is kept by status (feasible over a limit) and then by measured infeasibility.
//
// DETERMINISM. Which engine finishes first can depend on the machine's load. Every answer is
// verified the same way, so the objective is the same to the tolerance whoever wins, but on
// a model with several optimal vertices the vertex reported may differ between runs. That
// is the documented price of `algorithm=concurrent`; the single engines stay deterministic.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "nirnay/logging.hpp"
#include "nirnay/model.hpp"
#include "nirnay/solve_control.hpp"

namespace nirnay {

/// One engine in the race: a name for the log, and a run that honours the control it is
/// given and logs to the logger it is given.
struct Racer {
  std::string name;
  std::function<Solution(Logger& logger, SolveControl* control)> run;
};

/// Run every racer in its own thread; return the first optimal answer, or the best one.
/// `outer` (may be null) is the caller's control: interrupting it interrupts every racer.
[[nodiscard]] Solution race(const std::vector<Racer>& racers, Logger& logger,
                            SolveControl* outer);

}  // namespace nirnay

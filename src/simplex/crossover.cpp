// SPDX-License-Identifier: Apache-2.0
// NIRNAY - crossover. See crossover.hpp for the references and the contract.

#include "simplex/crossover.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace nirnay {
namespace {

/// Distance from `value` to the nearer of two bounds; infinite when neither exists.
double distance_to_bounds(double value, double lower, double upper) {
  double distance = std::numeric_limits<double>::infinity();
  if (is_finite_bound(lower)) distance = std::min(distance, value - lower);
  if (is_finite_bound(upper)) distance = std::min(distance, upper - value);
  return std::max(distance, 0.0);
}

/// The nonbasic status a variable at `value` takes: the nearer finite bound, or free.
BasisStatus nearest_bound(double value, double lower, double upper) {
  if (lower == upper) return BasisStatus::kFixed;
  const bool has_lower = is_finite_bound(lower);
  const bool has_upper = is_finite_bound(upper);
  if (has_lower && has_upper) {
    return value - lower <= upper - value ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
  }
  if (has_lower) return BasisStatus::kAtLower;
  if (has_upper) return BasisStatus::kAtUpper;
  return BasisStatus::kNonbasicFree;
}

struct Candidate {
  Index variable;   ///< j < n: structural j; j >= n: logical of row j - n
  double primal;    ///< distance to the nearest bound, relative to the value's scale
  double dual;      ///< |reduced cost| or |row price|, relative to its scale
  double score() const {
    // The El-Bakry/Tapia/Zhang indicator in its simplest form: a basic variable has a primal
    // distance that stays away from zero while its reduced cost goes to zero; a nonbasic one
    // the reverse. Free variables (infinite distance) score 1 and go first.
    if (std::isinf(primal)) return 2.0;
    const double total = primal + dual;
    return total > 0.0 ? primal / total : 0.5;
  }
};

}  // namespace

WarmStart identify_basis(const Model& model, const Solution& point) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  WarmStart warm;
  warm.col_status.assign(static_cast<std::size_t>(n), BasisStatus::kAtLower);
  warm.row_status.assign(static_cast<std::size_t>(m), BasisStatus::kAtLower);
  if (point.col_value.size() != static_cast<std::size_t>(n)) return {};

  std::vector<double> activity = point.row_activity;
  if (activity.size() != static_cast<std::size_t>(m)) {
    activity.assign(static_cast<std::size_t>(m), 0.0);
    if (m > 0) model.matrix.multiply(point.col_value.data(), activity.data());
  }
  const auto dual_of = [](const std::vector<double>& v, std::size_t k) {
    return k < v.size() ? std::fabs(v[k]) : 0.0;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<std::size_t>(n + m));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = point.col_value[u];
    const double lower = model.col_lower[u];
    const double upper = model.col_upper[u];
    warm.col_status[u] = nearest_bound(x, lower, upper);
    if (lower == upper) continue;  // a fixed column is never basic by choice
    const double scale = 1.0 + std::fabs(x);
    candidates.push_back({j, distance_to_bounds(x, lower, upper) / scale,
                          dual_of(point.col_dual, u) / (1.0 + std::fabs(model.col_cost[u]))});
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = activity[u];
    const double lower = model.row_lower[u];
    const double upper = model.row_upper[u];
    warm.row_status[u] = nearest_bound(a, lower, upper);
    const double scale = 1.0 + std::fabs(a);
    // An equality row's logical is fixed; it can still be basic (the simplex needs m basic
    // variables and a slack of zero width is legal in a basis), but only as a last resort.
    const double primal = lower == upper ? 0.0 : distance_to_bounds(a, lower, upper) / scale;
    candidates.push_back({n + i, primal, dual_of(point.row_dual, u)});
  }

  // The m most "basic-looking" variables, ties broken towards logicals (a slack basis is
  // always nonsingular, so leaning on logicals keeps the identified basis factorizable) and
  // then by index, so the result is deterministic.
  std::stable_sort(candidates.begin(), candidates.end(),
                   [n](const Candidate& a, const Candidate& b) {
                     const double sa = a.score();
                     const double sb = b.score();
                     if (sa != sb) return sa > sb;
                     if (a.primal != b.primal) return a.primal > b.primal;
                     return (a.variable >= n) > (b.variable >= n);
                   });
  const std::size_t basic = std::min(candidates.size(), static_cast<std::size_t>(m));
  for (std::size_t k = 0; k < basic; ++k) {
    const Index v = candidates[k].variable;
    if (v < n) {
      warm.col_status[static_cast<std::size_t>(v)] = BasisStatus::kBasic;
    } else {
      warm.row_status[static_cast<std::size_t>(v - n)] = BasisStatus::kBasic;
    }
  }
  return warm;
}

bool crossover_wanted(const Solution& approximate, const Model& model, const Options& options) {
  const std::string mode = options.get_string("crossover");
  if (mode == "off") return false;
  if (mode == "on") return true;
  return approximate.status != SolveStatus::kOptimal ||
         model.num_rows() <= options.get_int("crossover_auto_max_rows");
}

void crossover(Solution* approximate, const Model& model, const Options& options,
               Logger& logger, const Timer& timer, SolveControl* control) {
  if (!claims_a_point(approximate->status) || approximate->status == SolveStatus::kUnbounded) {
    return;
  }
  if (!crossover_wanted(*approximate, model, options)) return;
  if (!approximate->col_status.empty() &&
      std::any_of(approximate->col_status.begin(), approximate->col_status.end(),
                  [](BasisStatus s) { return s == BasisStatus::kBasic; })) {
    return;  // already a basis: the simplex produced it
  }

  double budget = options.get_double("crossover_max_seconds");
  if (options.get_string("crossover") == "auto" &&
      approximate->status == SolveStatus::kOptimal) {
    // The answer is already proved; the vertex is a refinement, so under `auto` it may cost
    // at most as long again as the solve did. Measured on the 5,000-row random family: PDHG
    // proves the optimum in about 2 s and the dual simplex then spends its whole 60 s cap
    // failing to finish from the identified basis - a 30x surcharge for nothing.
    budget = std::min(budget, std::max(1.0, timer.elapsed_seconds()));
  }
  const double time_limit = options.get_double("time_limit");
  if (time_limit > 0.0 && std::isfinite(time_limit)) {
    budget = std::min(budget, time_limit - timer.elapsed_seconds());
  }
  if (budget <= 0.0) {
    approximate->message += "; no time left for crossover";
    return;
  }

  const WarmStart warm = identify_basis(model, *approximate);
  if (warm.empty()) return;
  Options simplex_options = options;
  simplex_options.set_double("time_limit", budget);
  simplex_options.set_int("iteration_limit", -1);
  logger.info("Crossover: identifying a basis from the {} point and finishing with the dual "
              "simplex ({:.0f} s at most)",
              approximate->algorithm, budget);
  Solution vertex = solve_dual_simplex(model, simplex_options, logger, &warm, control);

  if (vertex.status != SolveStatus::kOptimal) {
    approximate->message += fmt::format(
        "; crossover did not reach an optimal basis ({} after {} simplex iterations), so the "
        "{} answer stands",
        to_string(vertex.status), vertex.iterations, approximate->algorithm);
    logger.info("Crossover: {} after {} iterations; keeping the {} answer",
                to_string(vertex.status), vertex.iterations, approximate->algorithm);
    return;
  }
  const double moved = std::fabs(vertex.objective - approximate->objective) /
                       std::max(1.0, std::fabs(approximate->objective));
  logger.info("Crossover: optimal basis after {} simplex iterations; objective moved by {:.2e} "
              "relative",
              vertex.iterations, moved);
  vertex.crossover_iterations = vertex.iterations;
  vertex.polish_iterations = approximate->polish_iterations;
  vertex.iterations += approximate->iterations;
  vertex.algorithm = approximate->algorithm + "+crossover";
  vertex.message = fmt::format(
      "{}; crossover to an optimal basis in {} simplex iterations (objective moved {:.2e} "
      "relative)",
      approximate->message, vertex.crossover_iterations, moved);
  *approximate = std::move(vertex);
}

}  // namespace nirnay

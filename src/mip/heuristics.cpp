// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the feasibility pump and RINS. See heuristics.hpp for the references and why.

#include "mip/heuristics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "nirnay/mip.hpp"
#include "nirnay/timer.hpp"
#include "nirnay/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace nirnay::mip {
namespace {

std::vector<Index> integer_columns_of(const Model& model) {
  std::vector<Index> columns;
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) columns.push_back(j);
  }
  return columns;
}

/// The nearest integer to `value` inside the column's bounds.
double round_into_bounds(double value, double lower, double upper) {
  double r = std::round(value);
  if (is_finite_bound(lower)) r = std::max(r, std::ceil(lower - 1e-9));
  if (is_finite_bound(upper)) r = std::min(r, std::floor(upper + 1e-9));
  return r;
}

/// `x` with every integer column rounded into its bounds; continuous columns kept.
std::vector<double> rounded(const Model& model, const std::vector<Index>& integers,
                            const std::vector<double>& x) {
  std::vector<double> r = x;
  for (const Index j : integers) {
    const auto u = static_cast<std::size_t>(j);
    r[u] = round_into_bounds(x[u], model.col_lower[u], model.col_upper[u]);
  }
  return r;
}

Options silent_lp_options(const Options& options, double seconds) {
  Options lp = options;
  lp.set_bool("log_to_console", false);
  lp.set_double("time_limit", std::max(seconds, 0.01));
  lp.set_int("iteration_limit", -1);
  return lp;
}

}  // namespace

int feasibility_pump(const Model& model, const std::vector<double>& start, const Options& options,
                     int max_rounds, double seconds, const OfferIncumbent& offer,
                     SolveControl* control) {
  const std::vector<Index> integers = integer_columns_of(model);
  const auto n = static_cast<std::size_t>(model.num_cols());
  if (integers.empty() || model.has_quadratic_objective() || start.size() != n) return 0;

  const Timer timer;
  Logger silent(nullptr);
  // The distance LP: the model's own rows and bounds, a new objective each round.
  Model distance = model;
  distance.sense = ObjSense::kMinimize;
  distance.objective_offset = 0.0;
  for (const Index j : integers) distance.col_type[static_cast<std::size_t>(j)] = VarType::kContinuous;

  std::mt19937 rng(static_cast<unsigned>(options.get_int("random_seed")) + 17u);
  std::vector<double> x = start;
  std::vector<double> previous;
  WarmStart warm;
  int solved = 0;
  for (int round = 0; round < max_rounds; ++round) {
    if (timer.elapsed_seconds() > seconds) break;
    if (control != nullptr && control->is_interrupted()) break;

    std::vector<double> target = rounded(model, integers, x);
    // The rounding itself, with the LP point's continuous part: often feasible outright.
    if (offer(target)) return solved;

    if (target == previous) {
      // A cycle: the projection returned to the same rounding. Flip the integer columns the
      // LP point disagrees with most (FGL 2005, section 3), a random number of them.
      std::vector<std::pair<double, Index>> disagreement;
      for (const Index j : integers) {
        const auto u = static_cast<std::size_t>(j);
        disagreement.emplace_back(std::fabs(x[u] - target[u]), j);
      }
      std::sort(disagreement.begin(), disagreement.end(), std::greater<>());
      const std::size_t flips =
          std::min<std::size_t>(disagreement.size(), 10 + rng() % 21);
      for (std::size_t k = 0; k < flips; ++k) {
        const auto u = static_cast<std::size_t>(disagreement[k].second);
        const double lower = model.col_lower[u];
        const double upper = model.col_upper[u];
        const double step = x[u] >= target[u] ? 1.0 : -1.0;
        double flipped = target[u] + (step != 0.0 ? step : 1.0);
        if ((is_finite_bound(upper) && flipped > upper) ||
            (is_finite_bound(lower) && flipped < lower)) {
          flipped = target[u] - step;
        }
        target[u] = round_into_bounds(flipped, lower, upper);
      }
    }
    previous = target;

    // Project: min sum over integer columns at a bound of the distance to that bound.
    std::fill(distance.col_cost.begin(), distance.col_cost.end(), 0.0);
    bool any_weight = false;
    for (const Index j : integers) {
      const auto u = static_cast<std::size_t>(j);
      const double lower = model.col_lower[u];
      const double upper = model.col_upper[u];
      if (is_finite_bound(lower) && target[u] <= lower + 0.5) {
        distance.col_cost[u] = 1.0;
        any_weight = true;
      } else if (is_finite_bound(upper) && target[u] >= upper - 0.5) {
        distance.col_cost[u] = -1.0;
        any_weight = true;
      }
    }
    if (!any_weight) break;  // only interior general integers: nothing this form can pull on
    const Options lp = silent_lp_options(options, seconds - timer.elapsed_seconds());
    const Solution projection =
        solve_dual_simplex(distance, lp, silent, warm.empty() ? nullptr : &warm, control);
    ++solved;
    if (projection.status != SolveStatus::kOptimal) break;
    warm = WarmStart{projection.col_status, projection.row_status};
    x = projection.col_value;
    bool integral = true;
    for (const Index j : integers) {
      const double v = x[static_cast<std::size_t>(j)];
      if (std::fabs(v - std::round(v)) > tol::kIntegrality) {
        integral = false;
        break;
      }
    }
    // Distance zero: the projection IS integral and LP-feasible. Snap and offer it.
    if (integral && offer(rounded(model, integers, x))) return solved;
  }
  return solved;
}

bool rins(const Model& model, const std::vector<double>& incumbent,
          const std::vector<double>& relaxation, const Options& options, Count node_limit,
          double seconds, double min_fixed_fraction, const OfferIncumbent& offer,
          SolveControl* control) {
  const std::vector<Index> integers = integer_columns_of(model);
  const auto n = static_cast<std::size_t>(model.num_cols());
  if (integers.empty() || model.has_quadratic_objective() || incumbent.size() != n ||
      relaxation.size() != n || seconds <= 0.0) {
    return false;
  }

  Model sub = model;
  std::size_t fixed = 0;
  for (const Index j : integers) {
    const auto u = static_cast<std::size_t>(j);
    if (std::fabs(incumbent[u] - relaxation[u]) <= tol::kIntegrality) {
      const double value = std::round(incumbent[u]);
      sub.col_lower[u] = value;
      sub.col_upper[u] = value;
      ++fixed;
    }
  }
  const double fraction = static_cast<double>(fixed) / static_cast<double>(integers.size());
  if (fraction < min_fixed_fraction || fixed == integers.size()) return false;

  // The objective cutoff as a row: only strictly better points are worth the search.
  double incumbent_objective = 0.0;
  for (std::size_t j = 0; j < n; ++j) incumbent_objective += model.col_cost[j] * incumbent[j];
  const double margin = 1e-6 * std::max(1.0, std::fabs(incumbent_objective));
  const Index cutoff_row = sub.num_rows();
  SparseMatrix grown(cutoff_row + 1, model.num_cols());
  grown.reserve(static_cast<std::size_t>(model.num_nonzeros()) + n);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) grown.add_entry(column.rows[k], j, column.values[k]);
    const double c = model.col_cost[static_cast<std::size_t>(j)];
    if (c != 0.0) grown.add_entry(cutoff_row, j, c);
  }
  grown.finalize();
  sub.matrix = std::move(grown);
  sub.resize_rows(cutoff_row + 1);
  const auto r = static_cast<std::size_t>(cutoff_row);
  if (model.sense == ObjSense::kMaximize) {
    sub.row_lower[r] = incumbent_objective + margin;
    sub.row_upper[r] = kInfinity;
  } else {
    sub.row_lower[r] = -kInfinity;
    sub.row_upper[r] = incumbent_objective - margin;
  }
  if (!sub.row_names.empty()) sub.row_names[r] = "rins_cutoff";

  Options sub_options = options;
  sub_options.set_bool("log_to_console", false);
  sub_options.set_bool("mip_heuristics", false);  // no RINS inside RINS
  sub_options.set_bool("enable_root_cuts", false);
  sub_options.set_int("node_limit", node_limit);
  sub_options.set_double("time_limit", seconds);
  Logger silent(nullptr);
  const Solution found = solve_branch_and_bound(sub, sub_options, silent, control);
  if (found.col_value.size() != n) return false;
  if (found.status != SolveStatus::kOptimal && found.status != SolveStatus::kFeasible &&
      found.status != SolveStatus::kNodeLimit && found.status != SolveStatus::kTimeLimit) {
    return false;
  }
  return offer(found.col_value);
}

}  // namespace nirnay::mip

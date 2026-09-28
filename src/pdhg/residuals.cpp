// SPDX-License-Identifier: Apache-2.0
// NIRNAY - PDHG residuals in original units. See residuals.hpp.

#include "pdhg/residuals.hpp"

#include <cmath>
#include <limits>

namespace nirnay::pdhg {
namespace {

double euclidean_norm(const std::vector<double>& v) {
  double total = 0.0;
  for (const double value : v) total += value * value;
  return std::sqrt(total);
}

/// Fill the objective-gap fields from the two objectives, identically for both measurements.
void finish_gap(Residuals* r) {
  const double absolute_gap = std::fabs(r->primal_objective - r->dual_objective);
  r->gap = absolute_gap / (1.0 + std::fabs(r->primal_objective) + std::fabs(r->dual_objective));
  r->gap_as_verified = absolute_gap / std::max(1.0, std::fabs(r->primal_objective));
}

}  // namespace

Problem make_problem(const Model& model) {
  Problem problem;
  problem.model = &model;
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();
  problem.cost.resize(static_cast<std::size_t>(cols));
  for (Index j = 0; j < cols; ++j) {
    problem.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  }
  problem.cost_norm = euclidean_norm(problem.cost);
  double bound_square = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double b = is_finite_bound(model.row_lower[u])
                         ? model.row_lower[u]
                         : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
    bound_square += b * b;
  }
  problem.bound_norm = std::sqrt(bound_square);
  return problem;
}

Residuals evaluate(const Problem& problem, const std::vector<double>& x,
                   const std::vector<double>& y, std::vector<double>* activity,
                   std::vector<double>* reduced) {
  const Model& model = *problem.model;
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();

  Residuals r;

  // ---- Primal: how far Ax falls outside the row bounds. x is projected every iteration,
  // so the column bounds hold by construction and contribute nothing.
  activity->assign(static_cast<std::size_t>(rows), 0.0);
  if (rows > 0) model.matrix.multiply(x.data(), activity->data());
  double primal_violation = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = (*activity)[u];
    double violation = 0.0;
    if (is_finite_bound(model.row_lower[u])) {
      violation = std::max(violation, model.row_lower[u] - a);
    }
    if (is_finite_bound(model.row_upper[u])) {
      violation = std::max(violation, a - model.row_upper[u]);
    }
    primal_violation += violation * violation;
  }
  r.absolute_primal = std::sqrt(primal_violation);
  r.primal = r.absolute_primal / (1.0 + problem.bound_norm);

  // ---- Dual: d = c + A'y. A component of d is only a violation where no bound can absorb
  // it, i.e. a positive reduced cost on a variable with no lower bound, or a negative one
  // on a variable with no upper bound.
  reduced->assign(static_cast<std::size_t>(cols), 0.0);
  for (Index j = 0; j < cols; ++j) {
    (*reduced)[static_cast<std::size_t>(j)] = problem.cost[static_cast<std::size_t>(j)];
  }
  if (rows > 0) model.matrix.transpose_multiply_add(y.data(), reduced->data());

  double dual_violation = 0.0;
  double bound_contribution = 0.0;
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = (*reduced)[u];
    if (d > 0.0) {
      if (is_finite_bound(model.col_lower[u])) {
        bound_contribution += d * model.col_lower[u];
      } else {
        dual_violation += d * d;
      }
    } else if (d < 0.0) {
      if (is_finite_bound(model.col_upper[u])) {
        bound_contribution += d * model.col_upper[u];
      } else {
        dual_violation += d * d;
      }
    }
  }

  // ---- Objectives. The dual objective is the Lagrangian bound:
  //   sum_j (d_j > 0 ? d_j l_j : d_j u_j)  -  sigma_C(y)
  double support = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double yi = y[u];
    if (yi > 0.0) {
      if (is_finite_bound(model.row_upper[u])) {
        support += yi * model.row_upper[u];
      } else {
        dual_violation += yi * yi;  // no bound to price against: the dual is infeasible
      }
    } else if (yi < 0.0) {
      if (is_finite_bound(model.row_lower[u])) {
        support += yi * model.row_lower[u];
      } else {
        dual_violation += yi * yi;
      }
    }
  }
  r.absolute_dual = std::sqrt(dual_violation);
  r.dual = r.absolute_dual / (1.0 + problem.cost_norm);

  // Complementary slackness, in the product form the verifier uses.
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (model.row_lower[u] == model.row_upper[u]) continue;  // equality: always tight
    const double lower_slack = is_finite_bound(model.row_lower[u])
                                   ? (*activity)[u] - model.row_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.row_upper[u])
                                   ? model.row_upper[u] - (*activity)[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity =
        std::max(r.complementarity, std::fabs(y[u]) * std::min(lower_slack, upper_slack));
  }
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] == model.col_upper[u]) continue;  // fixed column
    const double lower_slack = is_finite_bound(model.col_lower[u])
                                   ? x[u] - model.col_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.col_upper[u])
                                   ? model.col_upper[u] - x[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity = std::max(r.complementarity,
                                 std::fabs((*reduced)[u]) * std::min(lower_slack, upper_slack));
  }

  double primal_objective = 0.0;
  for (Index j = 0; j < cols; ++j) {
    primal_objective +=
        problem.cost[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  r.primal_objective = primal_objective;
  r.dual_objective = bound_contribution - support;
  finish_gap(&r);
  return r;
}

Residuals from_sums(const Problem& problem, const KktSums& sums) {
  Residuals r;
  r.absolute_primal = std::sqrt(sums.primal_violation_squared);
  r.primal = r.absolute_primal / (1.0 + problem.bound_norm);
  r.absolute_dual = std::sqrt(sums.dual_violation_squared);
  r.dual = r.absolute_dual / (1.0 + problem.cost_norm);
  r.primal_objective = sums.primal_objective;
  r.dual_objective = sums.bound_contribution - sums.support;
  r.complementarity = sums.complementarity;
  finish_gap(&r);
  return r;
}

}  // namespace nirnay::pdhg

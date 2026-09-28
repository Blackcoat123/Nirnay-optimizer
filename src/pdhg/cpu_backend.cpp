// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the CPU backend for restarted PDHG. See backend.hpp for the contract and
// pdhg.cpp for the algorithm and its references.
//
// DETERMINISM. Two rules make the answer independent of the thread count, which is the
// promise the `threads` option makes everywhere in this project (#57):
//   1. every parallel loop writes each output element from exactly one thread - the sparse
//      products are row-by-row gathers over a row-compressed matrix, never scatters;
//   2. every reduction sums fixed-size chunks, each chunk in index order, and then the
//      chunk partials in chunk order. Which thread ran a chunk cannot change a bit of it.
// OpenMP's own `reduction` clause is not used, because its combination order is whatever
// the runtime chose that day.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "pdhg/backend.hpp"

#ifdef NIRNAY_HAVE_OPENMP
#include <omp.h>
#endif

namespace nirnay::pdhg {
namespace {

/// Elements per reduction chunk, and the size below which a loop is not worth forking.
constexpr std::size_t kChunk = 4096;
constexpr std::ptrdiff_t kParallelThreshold = 32768;

bool finite_bound(double v) {
  return is_finite_bound(v);
}

double project(double value, double lower, double upper) {
  if (finite_bound(lower) && value < lower) return lower;
  if (finite_bound(upper) && value > upper) return upper;
  return value;
}

/// y = A x for a row-compressed A: one thread per output row, so the result is the same at
/// any thread count.
void multiply(const CompressedView& a, const double* x, double* y) {
  const auto outer = static_cast<std::ptrdiff_t>(a.outer);
#ifdef NIRNAY_HAVE_OPENMP
#pragma omp parallel for schedule(static, 256) if (outer > kParallelThreshold)
#endif
  for (std::ptrdiff_t i = 0; i < outer; ++i) {
    double sum = 0.0;
    const Index end = a.starts[i + 1];
    for (Index k = a.starts[i]; k < end; ++k) {
      sum += a.values[k] * x[a.indices[k]];
    }
    y[i] = sum;
  }
}

std::size_t chunk_count(std::size_t size) {
  return (size + kChunk - 1) / kChunk;
}

/// Run `body(begin, end, partial)` over fixed chunks of [0, size) and fold the partials in
/// chunk order with `fold(total, partial)`.
template <typename Partial, typename Body, typename Fold>
Partial chunked(std::size_t size, Body body, Fold fold) {
  const std::size_t chunks = chunk_count(size);
  std::vector<Partial> partials(chunks);
  const auto count = static_cast<std::ptrdiff_t>(chunks);
  [[maybe_unused]] const bool parallel = static_cast<std::ptrdiff_t>(size) > kParallelThreshold;
#ifdef NIRNAY_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (std::ptrdiff_t c = 0; c < count; ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * kChunk;
    const std::size_t end = std::min(size, begin + kChunk);
    body(begin, end, &partials[static_cast<std::size_t>(c)]);
  }
  Partial total{};
  for (const Partial& partial : partials) fold(&total, partial);
  return total;
}

/// An element-wise pass with no reduction.
template <typename Body>
void for_each_index(std::size_t size, Body body) {
  const auto count = static_cast<std::ptrdiff_t>(size);
#ifdef NIRNAY_HAVE_OPENMP
#pragma omp parallel for schedule(static) if (count > kParallelThreshold)
#endif
  for (std::ptrdiff_t i = 0; i < count; ++i) body(static_cast<std::size_t>(i));
}

struct Movement {
  double dx = 0.0;
  double dy = 0.0;
  double cross = 0.0;
};

void fold_movement(Movement* total, const Movement& part) {
  total->dx += part.dx;
  total->dy += part.dy;
  total->cross += part.cross;
}

struct RowSums {
  double primal = 0.0;
  double dual = 0.0;
  double support = 0.0;
  double complementarity = 0.0;
};

struct ColumnSums {
  double dual = 0.0;
  double bound = 0.0;
  double objective = 0.0;
  double complementarity = 0.0;
};

/// std::max(a, b) with the host evaluate()'s treatment of a NaN product: it is skipped.
double max_skipping_nan(double current, double candidate) {
  return std::isnan(candidate) ? current : std::max(current, candidate);
}

class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(const ScaledProblem& problem)
      : p_(problem),
        n_(static_cast<std::size_t>(problem.cols)),
        m_(static_cast<std::size_t>(problem.rows)) {
    for (Slot* slot : {&iterate_, &output_, &anchor_, &best_}) allocate(slot);
    x_sum_.assign(n_, 0.0);
    aty_sum_.assign(n_, 0.0);
    y_sum_.assign(m_, 0.0);
    ax_sum_.assign(m_, 0.0);
  }

  [[nodiscard]] std::string name() const override { return "pdhg-cpu"; }

  [[nodiscard]] std::string description() const override {
#ifdef NIRNAY_HAVE_OPENMP
    return fmt::format("CPU, {} thread(s)", omp_get_max_threads());
#else
    return "CPU, 1 thread (no OpenMP in this build)";
#endif
  }

  [[nodiscard]] double spectral_norm(int iterations, unsigned seed) override {
    if (m_ == 0 || n_ == 0 || p_.by_rows.starts[p_.rows] == 0) return 1.0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> spread(-1.0, 1.0);
    std::vector<double> v(n_);
    for (double& value : v) value = spread(rng);
    std::vector<double> av(m_, 0.0);
    double norm = 0.0;
    for (int step = 0; step < iterations; ++step) {
      double length = 0.0;
      for (const double value : v) length += value * value;
      length = std::sqrt(length);
      if (length == 0.0) return 1.0;
      for (double& value : v) value /= length;
      multiply(p_.by_rows, v.data(), av.data());
      multiply(p_.by_columns, av.data(), v.data());
      double next = 0.0;
      for (const double value : v) next += value * value;
      norm = std::sqrt(std::sqrt(next));
    }
    return norm > 0.0 ? norm * 1.01 : 1.0;
  }

  void initialize(const std::vector<double>& x, const std::vector<double>& y) override {
    iterate_.x = x;
    iterate_.y = y;
    refresh_products(&iterate_);
    anchor_ = iterate_;
    best_ = iterate_;
    output_ = iterate_;
    clear_average();
  }

  StepMeasure trial_step(double tau, double sigma) override {
    const Movement primal = chunked<Movement>(
        n_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t j = begin; j < end; ++j) {
            const double next = project(iterate_.x[j] - tau * (p_.cost[j] + iterate_.aty[j]),
                                        p_.col_lower[j], p_.col_upper[j]);
            output_.x[j] = next;
            const double d = next - iterate_.x[j];
            out->dx += d * d;
          }
        },
        fold_movement);
    multiply(p_.by_rows, output_.x.data(), output_.ax.data());
    const Movement dual = chunked<Movement>(
        m_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t i = begin; i < end; ++i) {
            const double v = iterate_.y[i] + sigma * (2.0 * output_.ax[i] - iterate_.ax[i]);
            const double next = v - sigma * project(v / sigma, p_.row_lower[i], p_.row_upper[i]);
            output_.y[i] = next;
            const double d = next - iterate_.y[i];
            out->dy += d * d;
            out->cross += d * (output_.ax[i] - iterate_.ax[i]);
          }
        },
        fold_movement);
    return {primal.dx, dual.dy, dual.cross};
  }

  void accept_trial() override {
    std::swap(iterate_.x, output_.x);
    std::swap(iterate_.y, output_.y);
    std::swap(iterate_.ax, output_.ax);
    multiply(p_.by_columns, iterate_.y.data(), iterate_.aty.data());
    for_each_index(n_, [&](std::size_t j) {
      x_sum_[j] += iterate_.x[j];
      aty_sum_[j] += iterate_.aty[j];
    });
    for_each_index(m_, [&](std::size_t i) {
      y_sum_[i] += iterate_.y[i];
      ax_sum_[i] += iterate_.ax[i];
    });
    ++averaged_;
  }

  void halpern_steps(Count count, Count first_k, double tau, double sigma,
                     double reflection) override {
    for (Count s = 0; s < count; ++s) {
      const auto k = static_cast<double>(first_k + s);
      const double w = (k + 1.0) / (k + 2.0);
      if (s + 1 == count) {
        halpern_step<true>(w, tau, sigma, reflection);
      } else {
        halpern_step<false>(w, tau, sigma, reflection);
      }
    }
  }

  [[nodiscard]] StepMeasure fixed_point_measure() const override { return fixed_point_; }

  [[nodiscard]] KktSums measure(Point point) override {
    const View v = view(point);
    const RowSums rows = chunked<RowSums>(
        m_,
        [&](std::size_t begin, std::size_t end, RowSums* out) {
          for (std::size_t i = begin; i < end; ++i) measure_row(v, i, out);
        },
        [](RowSums* total, const RowSums& part) {
          total->primal += part.primal;
          total->dual += part.dual;
          total->support += part.support;
          total->complementarity = max_skipping_nan(total->complementarity, part.complementarity);
        });
    const ColumnSums columns = chunked<ColumnSums>(
        n_,
        [&](std::size_t begin, std::size_t end, ColumnSums* out) {
          for (std::size_t j = begin; j < end; ++j) measure_column(v, j, out);
        },
        [](ColumnSums* total, const ColumnSums& part) {
          total->dual += part.dual;
          total->bound += part.bound;
          total->objective += part.objective;
          total->complementarity = max_skipping_nan(total->complementarity, part.complementarity);
        });
    KktSums sums;
    sums.primal_violation_squared = rows.primal;
    sums.dual_violation_squared = columns.dual + rows.dual;
    sums.bound_contribution = columns.bound;
    sums.support = rows.support;
    sums.primal_objective = columns.objective;
    sums.complementarity = std::max(rows.complementarity, columns.complementarity);
    return sums;
  }

  [[nodiscard]] StepMeasure distance_to_anchor(Point point) override {
    const View v = view(point);
    const Movement primal = chunked<Movement>(
        n_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t j = begin; j < end; ++j) {
            const double d = v.factor * v.x[j] - anchor_.x[j];
            out->dx += d * d;
          }
        },
        fold_movement);
    const Movement dual = chunked<Movement>(
        m_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t i = begin; i < end; ++i) {
            const double d = v.factor * v.y[i] - anchor_.y[i];
            out->dy += d * d;
          }
        },
        fold_movement);
    return {primal.dx, dual.dy, 0.0};
  }

  void restart_to(Point point) override {
    if (point != Point::kIterate) copy_into(view(point), &iterate_);
    anchor_ = iterate_;
    clear_average();
  }

  void save_best(Point point) override { copy_into(view(point), &best_); }

  void download(Point point, std::vector<double>* x, std::vector<double>* y) override {
    const View v = view(point);
    x->resize(n_);
    y->resize(m_);
    for (std::size_t j = 0; j < n_; ++j) (*x)[j] = v.factor * v.x[j];
    for (std::size_t i = 0; i < m_; ++i) (*y)[i] = v.factor * v.y[i];
  }

  [[nodiscard]] Count averaged() const override { return averaged_; }

 private:
  /// A point and its two products. Carried together because the products are linear in the
  /// point, which is what keeps every measurement free of sparse products.
  struct Slot {
    std::vector<double> x;    ///< cols
    std::vector<double> aty;  ///< cols, Ahat' y
    std::vector<double> y;    ///< rows
    std::vector<double> ax;   ///< rows, Ahat x
  };

  /// Read-only pointers to a point; the average is its sums times `factor`.
  struct View {
    const double* x;
    const double* aty;
    const double* y;
    const double* ax;
    double factor;
  };

  void allocate(Slot* slot) const {
    slot->x.assign(n_, 0.0);
    slot->aty.assign(n_, 0.0);
    slot->y.assign(m_, 0.0);
    slot->ax.assign(m_, 0.0);
  }

  void refresh_products(Slot* slot) const {
    multiply(p_.by_rows, slot->x.data(), slot->ax.data());
    multiply(p_.by_columns, slot->y.data(), slot->aty.data());
  }

  void clear_average() {
    std::fill(x_sum_.begin(), x_sum_.end(), 0.0);
    std::fill(aty_sum_.begin(), aty_sum_.end(), 0.0);
    std::fill(y_sum_.begin(), y_sum_.end(), 0.0);
    std::fill(ax_sum_.begin(), ax_sum_.end(), 0.0);
    averaged_ = 0;
  }

  [[nodiscard]] View view(Point point) const {
    switch (point) {
      case Point::kIterate:
        return {iterate_.x.data(), iterate_.aty.data(), iterate_.y.data(), iterate_.ax.data(),
                1.0};
      case Point::kOutput:
        return {output_.x.data(), output_.aty.data(), output_.y.data(), output_.ax.data(), 1.0};
      case Point::kAverage:
        if (averaged_ > 0) {
          return {x_sum_.data(), aty_sum_.data(), y_sum_.data(), ax_sum_.data(),
                  1.0 / static_cast<double>(averaged_)};
        }
        return view(Point::kIterate);
      case Point::kBest:
        return {best_.x.data(), best_.aty.data(), best_.y.data(), best_.ax.data(), 1.0};
    }
    return view(Point::kIterate);
  }

  void copy_into(const View& v, Slot* slot) const {
    for_each_index(n_, [&](std::size_t j) {
      slot->x[j] = v.factor * v.x[j];
      slot->aty[j] = v.factor * v.aty[j];
    });
    for_each_index(m_, [&](std::size_t i) {
      slot->y[i] = v.factor * v.y[i];
      slot->ax[i] = v.factor * v.ax[i];
    });
  }

  template <bool kMeasure>
  void halpern_step(double w, double tau, double sigma, double rho) {
    // T(z), primal half, and (on the measured step) |x - x~|^2.
    const Movement primal = chunked<Movement>(
        n_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t j = begin; j < end; ++j) {
            const double next = project(iterate_.x[j] - tau * (p_.cost[j] + iterate_.aty[j]),
                                        p_.col_lower[j], p_.col_upper[j]);
            output_.x[j] = next;
            if constexpr (kMeasure) {
              const double d = iterate_.x[j] - next;
              out->dx += d * d;
            }
          }
        },
        fold_movement);
    multiply(p_.by_rows, output_.x.data(), output_.ax.data());
    // T(z), dual half; then the Halpern update of y and, by linearity, of Ahat x.
    const Movement dual = chunked<Movement>(
        m_,
        [&](std::size_t begin, std::size_t end, Movement* out) {
          for (std::size_t i = begin; i < end; ++i) {
            const double ax = iterate_.ax[i];
            const double axt = output_.ax[i];
            const double v = iterate_.y[i] + sigma * (2.0 * axt - ax);
            const double next = v - sigma * project(v / sigma, p_.row_lower[i], p_.row_upper[i]);
            output_.y[i] = next;
            if constexpr (kMeasure) {
              const double d = iterate_.y[i] - next;
              out->dy += d * d;
              out->cross += d * (ax - axt);
            }
            iterate_.y[i] =
                w * ((1.0 + rho) * next - rho * iterate_.y[i]) + (1.0 - w) * anchor_.y[i];
            iterate_.ax[i] = w * ((1.0 + rho) * axt - rho * ax) + (1.0 - w) * anchor_.ax[i];
          }
        },
        fold_movement);
    multiply(p_.by_columns, output_.y.data(), output_.aty.data());
    for_each_index(n_, [&](std::size_t j) {
      iterate_.x[j] =
          w * ((1.0 + rho) * output_.x[j] - rho * iterate_.x[j]) + (1.0 - w) * anchor_.x[j];
      iterate_.aty[j] =
          w * ((1.0 + rho) * output_.aty[j] - rho * iterate_.aty[j]) + (1.0 - w) * anchor_.aty[j];
    });
    if constexpr (kMeasure) fixed_point_ = {primal.dx, dual.dy, dual.cross};
  }

  void measure_row(const View& v, std::size_t i, RowSums* out) const {
    const double lower = p_.original_row_lower[i];
    const double upper = p_.original_row_upper[i];
    const double activity = v.factor * v.ax[i] / p_.row_scale[i];
    double violation = 0.0;
    if (finite_bound(lower)) violation = std::max(violation, lower - activity);
    if (finite_bound(upper)) violation = std::max(violation, activity - upper);
    out->primal += violation * violation;

    const double yi = v.factor * v.y[i] * p_.row_scale[i];
    if (yi > 0.0) {
      if (finite_bound(upper)) {
        out->support += yi * upper;
      } else {
        out->dual += yi * yi;
      }
    } else if (yi < 0.0) {
      if (finite_bound(lower)) {
        out->support += yi * lower;
      } else {
        out->dual += yi * yi;
      }
    }
    if (lower != upper) {
      const double lower_slack =
          finite_bound(lower) ? activity - lower : std::numeric_limits<double>::infinity();
      const double upper_slack =
          finite_bound(upper) ? upper - activity : std::numeric_limits<double>::infinity();
      out->complementarity = max_skipping_nan(out->complementarity,
                                              std::fabs(yi) * std::min(lower_slack, upper_slack));
    }
  }

  void measure_column(const View& v, std::size_t j, ColumnSums* out) const {
    const double lower = p_.original_col_lower[j];
    const double upper = p_.original_col_upper[j];
    const double xj = v.factor * v.x[j] * p_.col_scale[j];
    const double d = p_.original_cost[j] + v.factor * v.aty[j] / p_.col_scale[j];
    if (d > 0.0) {
      if (finite_bound(lower)) {
        out->bound += d * lower;
      } else {
        out->dual += d * d;
      }
    } else if (d < 0.0) {
      if (finite_bound(upper)) {
        out->bound += d * upper;
      } else {
        out->dual += d * d;
      }
    }
    if (lower != upper) {
      const double lower_slack =
          finite_bound(lower) ? xj - lower : std::numeric_limits<double>::infinity();
      const double upper_slack =
          finite_bound(upper) ? upper - xj : std::numeric_limits<double>::infinity();
      out->complementarity = max_skipping_nan(out->complementarity,
                                              std::fabs(d) * std::min(lower_slack, upper_slack));
    }
    out->objective += p_.original_cost[j] * xj;
  }

  const ScaledProblem p_;
  const std::size_t n_;
  const std::size_t m_;
  Slot iterate_;
  Slot output_;
  Slot anchor_;
  Slot best_;
  std::vector<double> x_sum_;
  std::vector<double> aty_sum_;
  std::vector<double> y_sum_;
  std::vector<double> ax_sum_;
  Count averaged_ = 0;
  StepMeasure fixed_point_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const ScaledProblem& problem) {
  return std::make_unique<CpuBackend>(problem);
}

#ifndef NIRNAY_ENABLE_CUDA
std::unique_ptr<Backend> make_gpu_backend(const ScaledProblem& /*problem*/, int /*device*/,
                                          std::string* why) {
  if (why != nullptr) *why = "this build has no CUDA backend compiled in";
  return nullptr;
}
#endif

}  // namespace nirnay::pdhg

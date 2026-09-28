// SPDX-License-Identifier: Apache-2.0
// NIRNAY - restarted primal-dual hybrid gradient for LP, on the CPU or on a GPU.
//
// References, all written from the papers. Per the project's provenance rule the source of
// PDLP, cuPDLP, cuPDLP-C, cuPDLPx, OR-Tools and HiGHS was NOT consulted.
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011. Algorithm 1 is the step below.
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, "Practical Large-Scale
//           Linear Programming using Primal-Dual Hybrid Gradient", NeurIPS 2021.
//           Section 3.1 adaptive step size, 3.2 primal weight, 4.3 restarts.
//   [cuPDLP] Lu & Yang, "cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid
//           Gradient for Linear Programming in Julia", arXiv:2311.12180.
//   [HPDHG] Lu & Yang, "Restarted Halpern PDHG for Linear Programming", 2024: the Halpern
//           anchoring, its reflected variant, and restarts on the fixed-point residual.
//   [Hal67] Halpern, "Fixed points of nonexpanding maps", Bull. AMS 73(6), 1967.
//
// WHY THIS ENGINE EXISTS. The revised simplex is sequential: every pivot depends on the one
// before it, so it does not parallelise onto a GPU and we do not claim it does. PDHG
// replaces factorization with repeated sparse matrix-vector products, has no serial
// dependency inside an iteration, and is therefore the engine that can go on a GPU. The
// arithmetic lives behind a Backend (backend.hpp): OpenMP on the CPU, CUDA on a GPU, one
// algorithm above both.
//
// FORMULATION. Everything is kept in the two-sided form the Model already carries, with no
// row splitting and no slack variables:
//
//     min_{x in X} max_y   c'x + y'Ax - sigma_C(y),    X = [l, u],  C = [rl, ru]
//
// where sigma_C is the support function of the row-bound box. The y update is then a
// proximal step on sigma_C, and Moreau's identity turns that into a projection onto C:
//
//     prox_{s sigma_C}(v) = v - s * proj_C(v / s)
//
// which handles equality rows, one-sided rows, range rows and free rows through one
// expression. y here is the NEGATIVE of the multiplier the simplex reports, because the
// Lagrangian adds y'Ax rather than subtracting it; the report flips it back, and
// test_pdhg.cpp pins the two engines against each other so the flip cannot silently invert.
//
// TWO SCHEMES, one stopping rule. `pdhg_method=adaptive` is [PDLP]: a step size adapted
// every iteration from the observed movement, restarts to the better of the current and
// the averaged iterate on a KKT-error criterion. `pdhg_method=halpern` is [HPDHG]: a
// CONSTANT step size just inside 1/||A||_2, each step anchored back towards the last
// restart point with weight 1/(k+2) (Halpern's iteration, [Hal67]), optionally reflected,
// and restarts on the decay of the fixed-point residual ||z - T(z)||_P. The Halpern scheme
// needs no per-iteration decision, which is what lets a GPU run a whole block of iterations
// without a single transfer. Whichever runs, the loop stops only where the HOST - not the
// backend - measures a point that meets the project's absolute standard.

#include "nirnay/pdhg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "nirnay/timer.hpp"
#include "nirnay/tolerances.hpp"

#include "../la/scaling.hpp"
#include "pdhg/backend.hpp"
#include "pdhg/residuals.hpp"

namespace nirnay::pdhg {
namespace {

constexpr int kRuizIterations = 10;
/// Power iterations for ||A||_2. The adaptive scheme only starts from the estimate and then
/// learns the step; the Halpern scheme lives on it, so it gets more.
constexpr int kPowerIterationsAdaptive = 30;
constexpr int kPowerIterationsHalpern = 80;
/// The Halpern scheme's constant step, as a fraction of 1/||A||_2 (already rounded up 1%).
constexpr double kHalpernStepFraction = 0.99;
/// At most one iteration-table line per this many seconds; the first and last always print.
constexpr double kLogIntervalSeconds = 1.0;

double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// The scaled problem and the row-compressed copy of its matrix, owned here for as long as
/// the backend holds views into them.
struct Prepared {
  Scaling scaling;
  CsrView rows;
  ScaledProblem view;
};

void prepare(const Model& model, const Problem& problem, Prepared* prepared) {
  prepared->scaling = build_scaling(model, problem.cost, kRuizIterations);
  prepared->rows.build(prepared->scaling.matrix);
  const Scaling& s = prepared->scaling;
  ScaledProblem& v = prepared->view;
  v.rows = model.num_rows();
  v.cols = model.num_cols();
  v.by_rows = {v.rows, v.cols, prepared->rows.row_starts().data(),
               prepared->rows.column_indices().data(), prepared->rows.values().data()};
  v.by_columns = {v.cols, v.rows, s.matrix.column_starts().data(), s.matrix.row_indices().data(),
                  s.matrix.values().data()};
  v.cost = s.cost.data();
  v.col_lower = s.col_lower.data();
  v.col_upper = s.col_upper.data();
  v.row_lower = s.row_lower.data();
  v.row_upper = s.row_upper.data();
  v.col_scale = s.column.data();
  v.row_scale = s.row.data();
  v.original_cost = problem.cost.data();
  v.original_col_lower = model.col_lower.data();
  v.original_col_upper = model.col_upper.data();
  v.original_row_lower = model.row_lower.data();
  v.original_row_upper = model.row_upper.data();
}

/// The iteration state the two schemes share, and the one evaluation step both run.
class Loop {
 public:
  Loop(const Model& model, const Options& options, Logger& logger, SolveControl* control,
       const Problem& problem, const Prepared& prepared, Backend& backend, const Timer& timer)
      : model_(model),
        logger_(logger),
        control_(control),
        problem_(problem),
        prepared_(prepared),
        backend_(backend),
        timer_(timer),
        sense_(model.sense_multiplier()),
        tolerance_(options.get_double("pdhg_tolerance")),
        time_limit_(options.get_double("time_limit")),
        stop_at_request_(options.get_bool("pdhg_stop_at_request")),
        use_restarts_(options.get_bool("pdhg_restart")),
        interval_(std::max<Count>(1, options.get_int("pdhg_check_interval"))) {
    const std::int64_t limit = options.get_int("iteration_limit");
    iteration_limit_ = limit < 0 ? 1000000 : static_cast<Count>(limit);
    best_.primal = best_.dual = best_.gap = std::numeric_limits<double>::infinity();
  }

  /// [PDLP]: adaptive step, averaged restarts on the KKT error.
  void run_adaptive(double spectral_norm) {
    double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
    const double eta_ceiling = 1.0e3 / std::max(spectral_norm, 1e-12);
    double restart_kkt = std::numeric_limits<double>::infinity();
    Count last_restart = 0;
    while (!out_of_budget()) {
      const double tau = eta / omega_;
      const double sigma = eta * omega_;
      const StepMeasure step = backend_.trial_step(tau, sigma);
      // The step is admissible while eta <= (movement) / (interaction), both measured on the
      // step just taken, so a rejected step costs one product and is retried smaller.
      const double movement = 0.5 * omega_ * step.dx_squared + 0.5 * step.dy_squared / omega_;
      const double interaction = std::fabs(step.cross);
      // Zero interaction means the step carried NO information about how large eta may
      // safely be - the normal transient while the primal is still pinned against a bound,
      // and again once the iterates converge. Treating it as an infinite limit lets eta
      // overflow; setting limit = eta halves eta every such iteration down to the floor.
      // The step is trivially admissible, so accept it and leave eta exactly where it was.
      const bool no_information = interaction <= 0.0;
      const double limit =
          no_information ? std::numeric_limits<double>::infinity() : movement / interaction;
      // The exponent floor of 2 keeps `shrink` positive on the very first step: at exponent
      // 1 it is exactly zero and eta would collapse to its floor on iteration zero (#179).
      const double exponent = static_cast<double>(std::max<Count>(2, iteration_ + 1));
      const double shrink = 1.0 - std::pow(exponent, -0.3);
      const double grow = 1.0 + std::pow(exponent, -0.6);
      const double proposed = std::min(shrink * limit, grow * eta);
      if (eta <= limit) {
        backend_.accept_trial();
        ++iteration_;
      }
      if (!no_information) eta = std::clamp(proposed, 1e-12, eta_ceiling);

      // Evaluate on the periodic tick, at the iteration limit, and ALSO the moment the
      // iterates stop moving: a small problem can converge in fewer steps than the tick.
      if (iteration_ == 0) continue;
      const bool tick = iteration_ % interval_ == 0 || iteration_ >= iteration_limit_;
      if (!tick && !no_information) continue;

      Point chosen = Point::kIterate;
      Residuals measured;
      if (check(Point::kIterate, /*with_average=*/true, &chosen, &measured)) return;

      if (use_restarts_) {
        const double kkt = measured.worst();
        const Count since = iteration_ - last_restart;
        const bool sufficient = kkt <= 0.2 * restart_kkt;
        const bool artificial =
            since >= std::max<Count>(interval_,
                                     static_cast<Count>(0.36 * static_cast<double>(iteration_)));
        if (sufficient || artificial) {
          restart(chosen);
          restart_kkt = kkt;
          last_restart = iteration_;
          logger_.verbose("restart {} at iteration {}: KKT {:.3e}, primal weight {:.3e}",
                          restarts_, iteration_, kkt, omega_);
        }
      }
    }
  }

  /// [HPDHG]: constant step, Halpern anchoring, restarts on the fixed-point residual.
  void run_halpern(double spectral_norm, double reflection) {
    double eta = kHalpernStepFraction / std::max(spectral_norm, 1e-12);
    Count k = 0;  // Halpern's counter since the last restart
    double epoch_residual = std::numeric_limits<double>::infinity();
    double previous_residual = std::numeric_limits<double>::infinity();
    double seconds_per_iteration = 0.0;
    while (!out_of_budget()) {
      Count batch = std::min(interval_, iteration_limit_ - iteration_);
      if (seconds_per_iteration > 0.0) {
        // Keep a block of iterations from running far past the clock on a huge model. The
        // comparison stays in double: an unlimited clock divided by a microsecond is far
        // outside what a Count can hold.
        const double affordable =
            (time_limit_ - timer_.elapsed_seconds()) / seconds_per_iteration;
        if (affordable < static_cast<double>(batch)) {
          batch = std::max<Count>(1, static_cast<Count>(std::max(affordable, 1.0)));
        }
      }
      const double tau = eta / omega_;
      const double sigma = eta * omega_;
      const double started = timer_.elapsed_seconds();
      backend_.halpern_steps(batch, k, tau, sigma, reflection);
      k += batch;
      iteration_ += batch;
      seconds_per_iteration = (timer_.elapsed_seconds() - started) / static_cast<double>(batch);

      const StepMeasure fp = backend_.fixed_point_measure();
      const double residual = std::sqrt(
          std::max(0.0, fp.dx_squared / tau + fp.dy_squared / sigma - 2.0 * fp.cross));

      Point chosen = Point::kOutput;
      Residuals measured;
      if (check(Point::kOutput, /*with_average=*/false, &chosen, &measured)) return;

      // A diverging run - a step the norm estimate made too large - shows as a residual that
      // is not a number or has grown by orders of magnitude over its epoch. Halve the step
      // and go back to the best point seen: the theory needs eta < 1/||A||, and the estimate
      // approaches ||A|| from below before it is rounded up.
      const bool diverging = !std::isfinite(residual) || !std::isfinite(measured.worst()) ||
                             (std::isfinite(epoch_residual) && residual > 1e3 * epoch_residual);
      if (diverging) {
        eta *= 0.5;
        backend_.restart_to(Point::kBest);
        k = 0;
        epoch_residual = previous_residual = std::numeric_limits<double>::infinity();
        logger_.warning(
            "PDHG: the fixed-point residual diverged at iteration {}; halving the step to "
            "{:.3e} and restarting from the best point",
            iteration_, eta);
        continue;
      }
      if (!std::isfinite(epoch_residual)) epoch_residual = residual;

      if (use_restarts_) {
        const bool sufficient = residual <= 0.2 * epoch_residual;
        const bool necessary = residual <= 0.8 * epoch_residual && residual > previous_residual;
        const bool artificial = k >= static_cast<Count>(0.36 * static_cast<double>(iteration_));
        if (sufficient || necessary || artificial) {
          restart(Point::kOutput);
          k = 0;
          epoch_residual = residual;
          previous_residual = std::numeric_limits<double>::infinity();
          logger_.verbose(
              "restart {} at iteration {}: fixed-point residual {:.3e}, primal weight {:.3e}",
              restarts_, iteration_, residual, omega_);
          continue;
        }
      }
      previous_residual = residual;
    }
  }

  // ---- What the report needs -------------------------------------------------------------
  [[nodiscard]] bool converged() const { return converged_; }
  [[nodiscard]] bool interrupted() const { return interrupted_; }
  [[nodiscard]] const std::string& failure() const { return failure_; }
  [[nodiscard]] Count iterations() const { return iteration_; }
  [[nodiscard]] Count restarts() const { return restarts_; }

  /// The point to report, in original units: the one that passed, or the best seen.
  void final_point(std::vector<double>* x, std::vector<double>* y) {
    if (converged_) {
      *x = final_x_;
      *y = final_y_;
      return;
    }
    download_unscaled(Point::kBest, x, y);
  }

 private:
  [[nodiscard]] bool out_of_budget() const {
    return iteration_ >= iteration_limit_ || timer_.elapsed_seconds() > time_limit_;
  }

  void download_unscaled(Point point, std::vector<double>* x, std::vector<double>* y) {
    backend_.download(point, x, y);
    const Scaling& s = prepared_.scaling;
    for (std::size_t j = 0; j < x->size(); ++j) (*x)[j] *= s.column[j];
    for (std::size_t i = 0; i < y->size(); ++i) (*y)[i] *= s.row[i];
  }

  /// Measure, keep the best, log, poll, and test for convergence. Returns true to stop.
  bool check(Point current_point, bool with_average, Point* chosen_point, Residuals* chosen) {
    failure_ = backend_.failure();
    if (!failure_.empty()) return true;
    const Residuals current = from_sums(problem_, backend_.measure(current_point));
    *chosen_point = current_point;
    *chosen = current;
    // [PDLP] restarts to whichever of the running average and the current iterate has the
    // better KKT error, so both are measured and the better one is carried forward.
    if (with_average && backend_.averaged() > 0) {
      const Residuals average = from_sums(problem_, backend_.measure(Point::kAverage));
      if (average.worst() < current.worst()) {
        *chosen_point = Point::kAverage;
        *chosen = average;
      }
    }
    if (chosen->worst() < best_.worst()) {
      best_ = *chosen;
      backend_.save_best(*chosen_point);
    }

    // `+ model.objective_offset`: the logged objective has to be the quantity the final
    // line and the .sol file report, and presolve folds eliminated columns into the offset.
    const double objective = sense_ * chosen->primal_objective + model_.objective_offset;
    const double now = timer_.elapsed_seconds();
    if (!logged_table_ || now - last_log_ >= kLogIntervalSeconds) {
      if (!logged_table_) logger_.begin_iteration_table();
      logged_table_ = true;
      last_log_ = now;
      logger_.iteration(iteration_, objective, chosen->primal, chosen->dual, now);
    }
    if (control_ != nullptr) {
      Progress progress;
      progress.phase = SolvePhase::kLp;
      progress.iterations = iteration_;
      progress.objective = objective;
      progress.best_bound = sense_ * chosen->dual_objective + model_.objective_offset;
      progress.gap = chosen->gap_as_verified;
      progress.elapsed_seconds = now;
      if (control_->poll(progress)) {
        interrupted_ = true;
        return true;
      }
    }

    // THE STOPPING TEST AND THE REPORTING TEST ARE THE SAME TEST BY DEFAULT (#179): a point
    // that met the relative request but not the absolute standard would be downgraded to
    // `feasible` after the loop stopped on it, wasting the iterations that would have
    // reached the standard. `pdhg_stop_at_request` (#180) is the opt-out for the caller who
    // wants the cheap answer; it keeps absolute primal feasibility, and the report still
    // refuses `optimal` unless the full standard is met.
    const auto passes = [&](const Residuals& r) {
      return r.meets_request(tolerance_) && (stop_at_request_ || r.meets_project_standard());
    };
    if (!passes(*chosen)) return false;

    // The backend's measurement says stop. The HOST decides: the point is brought back,
    // unscaled, and re-measured against the original model with fresh products - the same
    // measurement the status is set from. A disagreement in the last bits keeps iterating.
    std::vector<double> x;
    std::vector<double> y;
    download_unscaled(*chosen_point, &x, &y);
    std::vector<double> activity;
    std::vector<double> reduced;
    const Residuals exact = evaluate(problem_, x, y, &activity, &reduced);
    if (!passes(exact)) {
      logger_.verbose(
          "PDHG: the backend measured convergence at iteration {} but the host measurement "
          "does not confirm it (primal {:.3e}, dual {:.3e}); continuing",
          iteration_, exact.absolute_primal, exact.absolute_dual);
      return false;
    }
    if (last_log_ != now) {
      logger_.iteration(iteration_, objective, chosen->primal, chosen->dual, now);
    }
    final_x_ = std::move(x);
    final_y_ = std::move(y);
    converged_ = true;
    return true;
  }

  /// Restart at `point` and move the primal weight towards the observed ratio of dual to
  /// primal movement since the last restart ([PDLP] section 3.2, theta = 0.5).
  void restart(Point point) {
    const StepMeasure distance = backend_.distance_to_anchor(point);
    const double dx = std::sqrt(distance.dx_squared);
    const double dy = std::sqrt(distance.dy_squared);
    if (dx > 1e-12 && dy > 1e-12) {
      const double theta = 0.5;
      omega_ = std::exp(theta * std::log(dy / dx) + (1.0 - theta) * std::log(omega_));
      omega_ = std::clamp(omega_, 1e-6, 1e6);
    }
    backend_.restart_to(point);
    ++restarts_;
  }

  const Model& model_;
  Logger& logger_;
  SolveControl* control_;
  const Problem& problem_;
  const Prepared& prepared_;
  Backend& backend_;
  const Timer& timer_;
  const double sense_;
  const double tolerance_;
  const double time_limit_;
  const bool stop_at_request_;
  const bool use_restarts_;
  const Count interval_;
  Count iteration_limit_ = 0;

  Count iteration_ = 0;
  Count restarts_ = 0;
  double omega_ = 1.0;  ///< primal weight
  Residuals best_;
  bool converged_ = false;
  bool interrupted_ = false;
  std::string failure_;
  bool logged_table_ = false;
  double last_log_ = 0.0;
  std::vector<double> final_x_;
  std::vector<double> final_y_;
};

/// The backend the options ask for, or the CPU with the reason logged.
std::unique_ptr<Backend> choose_backend(const Options& options, const ScaledProblem& view,
                                        Logger& logger) {
  if (options.get_bool("gpu")) {
    std::string why;
    std::unique_ptr<Backend> gpu =
        make_gpu_backend(view, static_cast<int>(options.get_int("gpu_device")), &why);
    if (gpu != nullptr) return gpu;
    logger.warning("--gpu requested but {}; running PDHG on the CPU", why);
  }
  return make_cpu_backend(view);
}

}  // namespace

Solution solve_pdhg(const Model& model, const Options& options, Logger& logger,
                    SolveControl* control) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cpu";

  const std::string problem_text = model.validate();
  if (!problem_text.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem_text;
    return solution;
  }

  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();
  const Problem problem = make_problem(model);

  // ---- Preconditioning, then the backend ---------------------------------------------------
  Prepared prepared;
  prepare(model, problem, &prepared);
  std::unique_ptr<Backend> backend = choose_backend(options, prepared.view, logger);
  solution.algorithm = backend->name();

  std::string method = options.get_string("pdhg_method");
  if (method == "auto") method = "halpern";
  const bool halpern = method == "halpern";
  const double spectral_norm =
      backend->spectral_norm(halpern ? kPowerIterationsHalpern : kPowerIterationsAdaptive,
                             static_cast<unsigned>(options.get_int("random_seed")) + 1u);
  const double reflection = options.get_double("pdhg_reflection");

  logger.info("Solving LP with restarted PDHG: {} rows, {} columns, {} nonzeros", rows, cols,
              model.num_nonzeros());
  logger.info("Backend: {}; scheme: {}{}", backend->description(), method,
              halpern ? fmt::format(" (reflection {:.2f})", reflection) : std::string());
  logger.info("Scaled matrix entries in [{:.3e}, {:.3e}], estimated ||A||_2 = {:.4e}",
              prepared.scaling.min_abs, prepared.scaling.max_abs, spectral_norm);
  logger.info("Target relative tolerance {:.1e}, restarts {}",
              options.get_double("pdhg_tolerance"),
              options.get_bool("pdhg_restart") ? "on" : "off");

  // Start at the projection of zero, which is the closest feasible point to the origin.
  std::vector<double> x0(static_cast<std::size_t>(cols), 0.0);
  const std::vector<double> y0(static_cast<std::size_t>(rows), 0.0);
  for (std::size_t j = 0; j < x0.size(); ++j) {
    x0[j] = project(0.0, prepared.scaling.col_lower[j], prepared.scaling.col_upper[j]);
  }
  backend->initialize(x0, y0);

  Loop loop(model, options, logger, control, problem, prepared, *backend, timer);
  if (halpern) {
    loop.run_halpern(spectral_norm, reflection);
  } else {
    loop.run_adaptive(spectral_norm);
  }

  if (!loop.failure().empty()) {
    // A device that failed mid-solve has nothing trustworthy left to hand back.
    solution.status = SolveStatus::kNumericalError;
    solution.message = fmt::format("the {} backend failed after {} iterations: {}",
                                   backend->name(), loop.iterations(), loop.failure());
    solution.iterations = loop.iterations();
    solution.solve_seconds = timer.elapsed_seconds();
    solution.col_value.clear();
    solution.row_activity.clear();
    logger.warning("{}", solution.message);
    return solution;
  }

  // ---- Report, from the HOST measurement of the point handed back -------------------------
  std::vector<double> best_x;
  std::vector<double> best_y;
  loop.final_point(&best_x, &best_y);
  std::vector<double> activity;
  std::vector<double> reduced;
  const Residuals final_residuals = evaluate(problem, best_x, best_y, &activity, &reduced);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_value[u] = best_x[u];
    solution.col_dual[u] = sense * reduced[u];
  }
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    // The Lagrangian above adds y'Ax, so the reported multiplier is the negation.
    solution.row_dual[u] = sense * (-best_y[u]);
  }
  solution.iterations = loop.iterations();
  solution.solve_seconds = timer.elapsed_seconds();
  const double time_limit = options.get_double("time_limit");
  const double tolerance = options.get_double("pdhg_tolerance");
  const bool stop_at_request = options.get_bool("pdhg_stop_at_request");

  // kOptimal is a claim that this point would survive tools/verify_solution.py, which
  // measures ABSOLUTE feasibility against the tolerances in tolerances.hpp. Meeting the
  // caller's RELATIVE tolerance is a weaker statement, and a point that stops on it but
  // misses the project standard is a usable answer with no optimality claim: kFeasible.
  const bool verifiable = loop.converged() && final_residuals.meets_project_standard();
  if (verifiable) {
    solution.status = SolveStatus::kOptimal;
    solution.message = fmt::format(
        "converged after {} iterations and {} restarts ({} scheme); absolute primal {:.3e}, "
        "dual {:.3e}, relative gap {:.3e}",
        loop.iterations(), loop.restarts(), method, final_residuals.absolute_primal,
        final_residuals.absolute_dual, final_residuals.gap_as_verified);
  } else if (loop.converged()) {
    solution.status = SolveStatus::kFeasible;
    solution.message = fmt::format(
        "met the requested relative tolerance {:.1e} after {} iterations, but NOT the "
        "absolute standard this project verifies against (primal {:.3e} vs {:.1e}, dual "
        "{:.3e} vs {:.1e}, relative gap {:.3e} vs {:.1e}). Reported as feasible, not "
        "optimal. {}",
        tolerance, loop.iterations(), final_residuals.absolute_primal, tol::kPrimalFeasibility,
        final_residuals.absolute_dual, tol::kDualFeasibility, final_residuals.gap_as_verified,
        tol::kDualityGap,
        stop_at_request ? "pdhg_stop_at_request is on, so this is the cheap answer that was "
                          "asked for; turn it off to run on to the standard"
                        : "Tighten --option pdhg_tolerance to close it");
  } else {
    // PDHG stopping short is the normal case, not an exception. Report the residuals it
    // actually reached rather than implying the point is optimal.
    solution.status = loop.interrupted()                     ? SolveStatus::kInterrupted
                      : timer.elapsed_seconds() > time_limit ? SolveStatus::kTimeLimit
                                                             : SolveStatus::kIterationLimit;
    solution.message = fmt::format(
        "{}at relative primal {:.3e}, dual {:.3e}, gap {:.3e} after {} iterations "
        "and {} restarts (target {:.1e})",
        loop.interrupted() ? "interrupted " : "stopped ", final_residuals.primal,
        final_residuals.dual, final_residuals.gap, loop.iterations(), loop.restarts(),
        tolerance);
  }

  // Only a verifiable point carries a dual bound. Anything else leaves it unknown, which is
  // the infinity on the unexplored side of the objective.
  if (verifiable) {
    solution.dual_bound = sense * final_residuals.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info("Status: {}   objective {:.10e}   iterations {}   restarts {}   time {:.3f}s",
              to_string(solution.status), solution.objective, solution.iterations,
              loop.restarts(), solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}",
              final_residuals.primal, final_residuals.dual, final_residuals.gap);
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace nirnay::pdhg

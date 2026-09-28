/* SPDX-License-Identifier: Apache-2.0
 *
 * NIRNAY - C API.
 *
 * A C89-compatible surface over the C++ core, for callers that cannot or will not link C++:
 * other languages' FFIs, and the Python bindings that sit on top of this rather than on the
 * C++ types directly. Nothing here exposes a C++ type, a template, or an exception - the
 * whole point is a boundary a `ctypes` script can cross.
 *
 * THE SHAPE IS DELIBERATELY FAMILIAR. create / set / solve / query, string-named options,
 * integer status codes. That is the surface every industrial solver presents, and matching
 * it is what makes this drop-in adoptable for someone with existing CPLEX or Gurobi calling
 * code. Per docs/CONTRIBUTING.md that is interface compatibility, not derivation: it is written from
 * the public shape those APIs document, and no solver source was read to produce it.
 *
 * ERRORS ARE RETURNED, NEVER THROWN. Every fallible call returns a nirnay_status. When one
 * is not NIRNAY_OK, nirnay_last_error() carries a human-readable reason for that thread.
 * A C caller cannot catch a C++ exception, so every entry point that could raise one wraps
 * its body and converts.
 *
 * OWNERSHIP. Handles returned by a *_create or *_solve function are owned by the caller and
 * must be released with the matching *_free. Every `const char*` returned by this API points
 * into storage owned by the library and is valid until the next call ON THE SAME THREAD that
 * could replace it - copy it if you need to keep it.
 *
 * THREADING. Handles are not internally synchronised: two threads must not touch one handle
 * at once. Distinct handles in distinct threads are fine, and the error string is
 * thread-local, so concurrent solves do not overwrite each other's diagnostics.
 */
#ifndef NIRNAY_H
#define NIRNAY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes ---------------------------------------------------------------------- */

typedef enum nirnay_status {
  NIRNAY_OK = 0,
  NIRNAY_ERROR_ARGUMENT = 1, /**< a null handle, or an index outside the model */
  NIRNAY_ERROR_IO = 2,       /**< the file could not be read or parsed */
  NIRNAY_ERROR_MODEL = 3,    /**< the model is not internally consistent; see last_error */
  NIRNAY_ERROR_OPTION = 4,   /**< no such option, or a value it will not accept */
  NIRNAY_ERROR_MEMORY = 5,   /**< allocation failed */
  NIRNAY_ERROR_INTERNAL = 6  /**< a C++ exception crossed the boundary and was converted */
} nirnay_status;

/** Mirrors nirnay::SolveStatus. Values are stable; new ones are appended. */
typedef enum nirnay_solve_status {
  NIRNAY_NOT_SOLVED = 0,
  NIRNAY_OPTIMAL = 1,
  NIRNAY_FEASIBLE = 2, /**< a usable point; optimality NOT proven */
  NIRNAY_INFEASIBLE = 3,
  NIRNAY_UNBOUNDED = 4,
  /** Not both feasible and bounded, without separating the two. Some first-order
   *  methods legitimately stop here; reporting it beats guessing which it was. */
  NIRNAY_INFEASIBLE_OR_UNBOUNDED = 10,
  NIRNAY_ITERATION_LIMIT = 5,
  NIRNAY_TIME_LIMIT = 6,
  NIRNAY_NODE_LIMIT = 7,
  NIRNAY_NUMERICAL_ERROR = 8,
  NIRNAY_MODEL_ERROR = 9,
  /** Stopped by a progress callback or nirnay_model_interrupt() (#223). Like the limit
   *  statuses, this normally carries a point - the incumbent, or the last feasible
   *  iterate. */
  NIRNAY_INTERRUPTED = 11
} nirnay_solve_status;

/* ---- Opaque handles -------------------------------------------------------------------- */

typedef struct nirnay_model nirnay_model;
typedef struct nirnay_options nirnay_options;
typedef struct nirnay_solution nirnay_solution;

/* ---- Library ---------------------------------------------------------------------------- */

/** Version string, e.g. "0.1.0 (abc1234, Release)". Never NULL. */
const char* nirnay_version(void);

/**
 * Human-readable reason for the most recent failing call ON THIS THREAD.
 *
 * Returns an empty string when nothing has failed. The pointer is valid until the next
 * failing call on this thread.
 */
const char* nirnay_last_error(void);

/** The infinity this API uses for absent bounds. Bounds at or beyond it are treated as free. */
double nirnay_infinity(void);

/* ---- Model ------------------------------------------------------------------------------ */

/** An empty minimisation model with no rows or columns. NULL only on allocation failure. */
nirnay_model* nirnay_model_create(void);
void nirnay_model_free(nirnay_model* model);

/**
 * Read a model from an MPS, QPS or LP file, choosing the reader by extension and content.
 *
 * On failure the handle is left unmodified and last_error carries the parser's message,
 * which names the line - these readers refuse ambiguous files rather than guessing, so a
 * failure here is usually a genuine defect in the file.
 */
nirnay_status nirnay_model_read(nirnay_model* model, const char* path);

/** 0 to minimise (the default), non-zero to maximise. */
nirnay_status nirnay_model_set_maximize(nirnay_model* model, int maximize);

/** Constant added to the objective. */
nirnay_status nirnay_model_set_objective_offset(nirnay_model* model, double offset);

/**
 * Append one column, returning its index through `index` when that is non-NULL.
 *
 * `name` may be NULL. Use +/- nirnay_infinity() for absent bounds. `is_integer` non-zero
 * makes this an integer column, which makes the model a MILP.
 */
nirnay_status nirnay_model_add_column(nirnay_model* model, double cost, double lower,
                                        double upper, int is_integer, const char* name,
                                        int* index);

/**
 * Append one row, returning its index through `index` when that is non-NULL.
 *
 * A range row is lower <= a'x <= upper; pass equal bounds for an equality, and an infinite
 * bound on one side for a one-sided inequality.
 */
nirnay_status nirnay_model_add_row(nirnay_model* model, double lower, double upper,
                                     const char* name, int* index);

/**
 * Set one constraint-matrix coefficient.
 *
 * Entries may be supplied in any order. Setting the same (row, column) twice REPLACES the
 * earlier value rather than summing it - summing is what MPS files mean by a repeated entry
 * and this API deliberately does not inherit that, because silently doubling a coefficient
 * is not a mistake a caller can see in the answer.
 *
 * A value of exactly zero removes the entry.
 */
nirnay_status nirnay_model_set_coefficient(nirnay_model* model, int row, int col,
                                             double value);

/**
 * Set one entry of the objective Hessian Q, making this a quadratic program.
 *
 * The objective is c'x + 0.5 x'Qx and Q is symmetric, so ONLY THE LOWER TRIANGLE is stored:
 * an entry (i, j) with i > j stands for both Q[i][j] and Q[j][i]. Passing (j, i) instead is
 * accepted and means the same thing. The 0.5 belongs to the objective, not to the value you
 * pass here - the same convention QPS files use.
 *
 * A non-convex Q is REFUSED at solve time with NIRNAY_MODEL_ERROR rather than solved to a
 * local point.
 */
nirnay_status nirnay_model_set_quadratic_coefficient(nirnay_model* model, int row, int col,
                                                       double value);

int nirnay_model_num_cols(const nirnay_model* model);
int nirnay_model_num_rows(const nirnay_model* model);
int nirnay_model_num_nonzeros(const nirnay_model* model);

/**
 * Check the model for internal consistency without solving it.
 *
 * Returns NIRNAY_OK when the model is well formed, NIRNAY_ERROR_MODEL otherwise with the
 * reason in last_error.
 */
nirnay_status nirnay_model_validate(const nirnay_model* model);

/* ---- Progress and interruption (#223) ---------------------------------------------------- */

/** Mirrors nirnay::Progress. Fields that do not apply to the reporting engine are 0. */
typedef struct nirnay_progress {
  int phase; /**< 0 presolve, 1 lp, 2 tree - mirrors nirnay::SolvePhase */
  long iterations;
  long nodes;
  long open_nodes;
  double objective;
  double best_bound;
  double gap;
  double elapsed_seconds;
} nirnay_progress;

/**
 * Called on the solving thread at a bounded rate (by iteration/node count, and never more
 * often than roughly every 100 ms) so a slow or chatty callback cannot materially slow the
 * solve down. Return non-zero to ask the solve to stop.
 */
typedef int (*nirnay_progress_callback)(const nirnay_progress* progress, void* user_data);

/**
 * Install a progress callback for every solve of `model` from here on, replacing any
 * previous one. A NULL `callback` removes it. `user_data` is opaque, passed back unchanged,
 * and its lifetime is the caller's responsibility - it must outlive every solve of this
 * model that could still invoke the callback.
 *
 * A solve whose callback returns non-zero stops with NIRNAY_INTERRUPTED and whatever point
 * (incumbent, or last feasible iterate) the engine was carrying at the time.
 */
nirnay_status nirnay_set_callback(nirnay_model* model, nirnay_progress_callback callback,
                                    void* user_data);

/**
 * Ask a solve of `model` running on ANOTHER THREAD to stop.
 *
 * Thread-safe, and NOT throttled the way the callback is: the solving thread sees it the
 * next time it checks, which is at least as often as it already checks its time limit.
 * Safe to call whether or not a solve is currently running; each call to nirnay_solve
 * starts with a clear flag, so calling this before or after a solve (rather than during
 * one) has no effect on the next one - it is a request to stop THIS solve, not a standing
 * instruction.
 */
nirnay_status nirnay_model_interrupt(nirnay_model* model);

/* ---- Options ---------------------------------------------------------------------------- */

/** Options preset to their documented defaults. Run `nirnay options` to list them. */
nirnay_options* nirnay_options_create(void);
void nirnay_options_free(nirnay_options* options);

nirnay_status nirnay_options_set_bool(nirnay_options* options, const char* name, int value);
nirnay_status nirnay_options_set_int(nirnay_options* options, const char* name, long value);
nirnay_status nirnay_options_set_double(nirnay_options* options, const char* name,
                                          double value);
nirnay_status nirnay_options_set_string(nirnay_options* options, const char* name,
                                          const char* value);

/* ---- Solve ------------------------------------------------------------------------------ */

/**
 * Solve, writing a newly allocated solution handle to `*solution`.
 *
 * `options` may be NULL for the defaults. The return value reports whether the CALL
 * succeeded, not what the solver concluded: a model proved infeasible returns NIRNAY_OK
 * with a solution whose status is NIRNAY_INFEASIBLE. Check both.
 */
nirnay_status nirnay_solve(const nirnay_model* model, const nirnay_options* options,
                             nirnay_solution** solution);

void nirnay_solution_free(nirnay_solution* solution);

nirnay_solve_status nirnay_solution_status(const nirnay_solution* solution);

/** Explanatory message from the solver. Empty when there is nothing to add. */
const char* nirnay_solution_message(const nirnay_solution* solution);

double nirnay_solution_objective(const nirnay_solution* solution);

/** Best proven bound. Equals the objective when optimality was proved. */
double nirnay_solution_dual_bound(const nirnay_solution* solution);

long nirnay_solution_iterations(const nirnay_solution* solution);
long nirnay_solution_nodes(const nirnay_solution* solution);
double nirnay_solution_seconds(const nirnay_solution* solution);

/**
 * MEASURED quality of the returned point, not asserted by the engine about itself.
 *
 * These are recomputed from the returned vectors before the solver reports anything, and
 * the dispatcher downgrades a status that disagrees with them. A caller writing its own
 * acceptance test should read these rather than trusting the status alone.
 */
double nirnay_solution_primal_infeasibility(const nirnay_solution* solution);
double nirnay_solution_dual_infeasibility(const nirnay_solution* solution);
double nirnay_solution_integrality_violation(const nirnay_solution* solution);

/**
 * Copy the primal column values into `values`, which must have room for `count` doubles.
 *
 * `count` must equal the model's column count; a mismatch returns NIRNAY_ERROR_ARGUMENT
 * rather than writing a partial vector, because a caller that has the dimension wrong is
 * about to misread every number it copies.
 */
nirnay_status nirnay_solution_col_values(const nirnay_solution* solution, double* values,
                                           int count);

/** Row activities a'x, same contract as nirnay_solution_col_values. */
nirnay_status nirnay_solution_row_activities(const nirnay_solution* solution, double* values,
                                               int count);

/** Row dual values (shadow prices), same contract. */
nirnay_status nirnay_solution_row_duals(const nirnay_solution* solution, double* values,
                                          int count);

/** Column reduced costs, same contract. */
nirnay_status nirnay_solution_col_duals(const nirnay_solution* solution, double* values,
                                          int count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIRNAY_H */

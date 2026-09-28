// SPDX-License-Identifier: Apache-2.0
// NIRNAY - C API implementation.
//
// Every entry point here does three things and nothing else: validate its arguments, call
// into the C++ core, and convert whatever comes back - including an exception - into a
// status code. There is no solver logic in this file and there must not be. The moment a
// decision lives only in the C wrapper, the C++ callers and the CLI stop agreeing with the
// bindings about what the library does.
//
// THE MATRIX IS ACCUMULATED, NOT BUILT. nirnay::SparseMatrix is frozen once finalized, but
// a C caller sets coefficients one at a time in whatever order suits it. So the handle keeps
// triplets in a map and materialises the matrix at solve time. The map also gives
// set_coefficient its REPLACE semantics for free, which is deliberately unlike the MPS
// reader's "a repeated entry is an error": a programmatic caller overwriting a cell is
// ordinary, whereas a file containing the same cell twice is a defect in the file.

#include "nirnay/nirnay.h"

#include <exception>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "nirnay/io.hpp"
#include "nirnay/model.hpp"
#include "nirnay/options.hpp"
#include "nirnay/solve_control.hpp"
#include "nirnay/version.hpp"

namespace {

// Thread-local so that concurrent solves cannot overwrite each other's diagnostics. A caller
// debugging one failing solve while another thread is busy would otherwise read a message
// belonging to a model it has never seen.
thread_local std::string g_error;

nirnay_status fail(nirnay_status code, std::string message) {
  g_error = std::move(message);
  return code;
}

nirnay_status ok() {
  g_error.clear();
  return NIRNAY_OK;
}

nirnay_solve_status to_c_status(nirnay::SolveStatus status) {
  switch (status) {
    case nirnay::SolveStatus::kNotSolved: return NIRNAY_NOT_SOLVED;
    case nirnay::SolveStatus::kOptimal: return NIRNAY_OPTIMAL;
    case nirnay::SolveStatus::kFeasible: return NIRNAY_FEASIBLE;
    case nirnay::SolveStatus::kInfeasible: return NIRNAY_INFEASIBLE;
    case nirnay::SolveStatus::kUnbounded: return NIRNAY_UNBOUNDED;
    case nirnay::SolveStatus::kInfeasibleOrUnbounded: return NIRNAY_INFEASIBLE_OR_UNBOUNDED;
    case nirnay::SolveStatus::kIterationLimit: return NIRNAY_ITERATION_LIMIT;
    case nirnay::SolveStatus::kTimeLimit: return NIRNAY_TIME_LIMIT;
    case nirnay::SolveStatus::kNodeLimit: return NIRNAY_NODE_LIMIT;
    case nirnay::SolveStatus::kInterrupted: return NIRNAY_INTERRUPTED;
    case nirnay::SolveStatus::kNumericalError: return NIRNAY_NUMERICAL_ERROR;
    case nirnay::SolveStatus::kModelError: return NIRNAY_MODEL_ERROR;
  }
  return NIRNAY_NOT_SOLVED;
}

/// Copy one of the solution's vectors out, refusing a size mismatch.
nirnay_status copy_vector(const std::vector<double>& source, double* destination, int count,
                           const char* what) {
  if (destination == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "destination is null");
  if (count < 0 || static_cast<std::size_t>(count) != source.size()) {
    return fail(NIRNAY_ERROR_ARGUMENT, std::string("wrong buffer size for ") + what +
                                            ": the solution has " +
                                            std::to_string(source.size()) + " entries, " +
                                            std::to_string(count) + " were offered");
  }
  for (std::size_t i = 0; i < source.size(); ++i) destination[i] = source[i];
  return ok();
}

/// Reject an option the registry does not have, or has with another type, BEFORE the typed
/// setter is reached.
///
/// This is not defensive style, it is a hard requirement of the boundary. The C++ typed
/// accessors treat an unknown name as a programmer error and call std::abort() - correct for
/// C++ code, where a typo is a bug the compiler nearly caught, and fatal here. A C caller
/// passing a mistyped option string would take down the host process, and the Python
/// bindings that will sit on this API would take the interpreter with them. Measured: the
/// test for this crashed with 0xC0000409 until the check existed.
nirnay_status check_option(const char* name, nirnay::OptionType wanted) {
  if (!nirnay::Options::exists(name)) {
    return fail(NIRNAY_ERROR_OPTION, std::string("unknown option '") + name +
                                          "'; run `nirnay options` for the list");
  }
  const nirnay::OptionSpec* spec = nirnay::Options::find_spec(name);
  if (spec != nullptr && spec->type != wanted) {
    return fail(NIRNAY_ERROR_OPTION,
                std::string("option '") + name + "' is not of the type this setter writes");
  }
  return NIRNAY_OK;
}

}  // namespace

// The handles. Defined here so the header can keep them opaque.
struct nirnay_model {
  nirnay::Model model;
  // (row, col) -> value, pending until the matrix is materialised.
  std::map<std::pair<int, int>, double> entries;
  std::map<std::pair<int, int>, double> quadratic;

  // #223. `mutable`: nirnay_solve takes a `const nirnay_model*` (it does not change the
  // model's mathematics), but the whole point of this member is that another thread calls
  // nirnay_model_interrupt() WHILE a solve is reading it - the callback/interrupt channel
  // is accessory state with its own internal synchronisation, not part of the model this
  // const-ness protects. See nirnay::SolveControl's own comment for the thread-safety
  // contract that makes this sound.
  mutable nirnay::SolveControl control;
};

struct nirnay_options {
  nirnay::Options options;
};

struct nirnay_solution {
  nirnay::Solution solution;
};

namespace {

/// Run `body`, converting any exception into a status code.
///
/// An exception crossing into C is undefined behaviour, so every escape route has to be
/// closed - including ones this file does not know about, hence the bare `catch (...)`.
///
/// A template rather than the macro this started as. A function-like macro splits its
/// argument on commas, so a body containing `entries[{row, col}]` arrives as two arguments
/// and the preprocessor rejects it - which it duly did, in four places.
template <typename Body>
nirnay_status guarded(Body&& body) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    return fail(NIRNAY_ERROR_MEMORY, "out of memory");
  } catch (const std::exception& error) {
    return fail(NIRNAY_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(NIRNAY_ERROR_INTERNAL, "an unknown exception crossed the C API");
  }
}

}  // namespace

extern "C" {

// ---- Library -----------------------------------------------------------------------------

const char* nirnay_version(void) {
  static const std::string version = nirnay::version_string();
  return version.c_str();
}

const char* nirnay_last_error(void) {
  return g_error.c_str();
}

double nirnay_infinity(void) {
  return nirnay::kInfinity;
}

// ---- Model -------------------------------------------------------------------------------

nirnay_model* nirnay_model_create(void) {
  try {
    return new nirnay_model();
  } catch (...) {
    return nullptr;
  }
}

void nirnay_model_free(nirnay_model* model) {
  delete model;
}

nirnay_status nirnay_model_read(nirnay_model* model, const char* path) {
  if (model == nullptr || path == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "model or path is null");
  }
  return guarded([&]() -> nirnay_status {
    nirnay::Model fresh;
    const nirnay::io::ReadResult result = nirnay::io::read_model(path, &fresh);
    if (!result.ok) return fail(NIRNAY_ERROR_IO, result.error);
    // Only on success. A half-populated handle after a failed read would be the worst of
    // both outcomes: the caller sees an error and still holds something solvable.
    model->model = std::move(fresh);
    model->entries.clear();
    model->quadratic.clear();
    return ok();
  });
}

nirnay_status nirnay_model_set_maximize(nirnay_model* model, int maximize) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  model->model.sense =
      maximize != 0 ? nirnay::ObjSense::kMaximize : nirnay::ObjSense::kMinimize;
  return ok();
}

nirnay_status nirnay_model_set_objective_offset(nirnay_model* model, double offset) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  model->model.objective_offset = offset;
  return ok();
}

nirnay_status nirnay_model_add_column(nirnay_model* model, double cost, double lower,
                                        double upper, int is_integer, const char* name,
                                        int* index) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> nirnay_status {
    nirnay::Model& m = model->model;
    const int position = static_cast<int>(m.col_cost.size());
    m.col_cost.push_back(cost);
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.col_type.push_back(is_integer != 0 ? nirnay::VarType::kInteger
                                         : nirnay::VarType::kContinuous);
    // Names are all-or-nothing: the solution writer and the verifier match on them, so a
    // model with names for only some columns would write a file neither can read back.
    // Columns added without one are given a positional name so the vector stays complete.
    m.col_names.push_back(name != nullptr ? std::string(name)
                                          : "C" + std::to_string(position + 1));
    if (index != nullptr) *index = position;
    return ok();
  });
}

nirnay_status nirnay_model_add_row(nirnay_model* model, double lower, double upper,
                                     const char* name, int* index) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> nirnay_status {
    nirnay::Model& m = model->model;
    const int position = static_cast<int>(m.row_lower.size());
    m.row_lower.push_back(lower);
    m.row_upper.push_back(upper);
    m.row_names.push_back(name != nullptr ? std::string(name)
                                          : "R" + std::to_string(position + 1));
    if (index != nullptr) *index = position;
    return ok();
  });
}

nirnay_status nirnay_model_set_coefficient(nirnay_model* model, int row, int col,
                                             double value) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  const nirnay::Model& m = model->model;
  if (row < 0 || row >= static_cast<int>(m.row_lower.size())) {
    return fail(NIRNAY_ERROR_ARGUMENT, "row " + std::to_string(row) +
                                            " is outside the model, which has " +
                                            std::to_string(m.row_lower.size()) + " row(s)");
  }
  if (col < 0 || col >= static_cast<int>(m.col_cost.size())) {
    return fail(NIRNAY_ERROR_ARGUMENT, "column " + std::to_string(col) +
                                            " is outside the model, which has " +
                                            std::to_string(m.col_cost.size()) + " column(s)");
  }
  return guarded([&]() -> nirnay_status {
    if (value == 0.0) {
      model->entries.erase({row, col});
    } else {
      model->entries[{row, col}] = value;
    }
    return ok();
  });
}

nirnay_status nirnay_model_set_quadratic_coefficient(nirnay_model* model, int row, int col,
                                                       double value) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  const int columns = static_cast<int>(model->model.col_cost.size());
  if (row < 0 || row >= columns || col < 0 || col >= columns) {
    return fail(NIRNAY_ERROR_ARGUMENT, "quadratic index (" + std::to_string(row) + ", " +
                                            std::to_string(col) +
                                            ") is outside the model, which has " +
                                            std::to_string(columns) + " column(s)");
  }
  return guarded([&]() -> nirnay_status {
    // Normalised to the lower triangle, so (i, j) and (j, i) name one entry of the symmetric
    // Q rather than two. A caller giving both would otherwise set the same coefficient twice
    // and, on a naive implementation, double it.
    const int lower_row = row > col ? row : col;
    const int lower_col = row > col ? col : row;
    if (value == 0.0) {
      model->quadratic.erase({lower_row, lower_col});
    } else {
      model->quadratic[{lower_row, lower_col}] = value;
    }
    return ok();
  });
}

nirnay_status nirnay_set_callback(nirnay_model* model, nirnay_progress_callback callback,
                                    void* user_data) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> nirnay_status {
    if (callback == nullptr) {
      model->control.set_callback(nullptr);
      return ok();
    }
    // Capture the C function pointer and its user_data by value; convert the C++ Progress
    // to the C nirnay_progress struct at the boundary, exactly the shape the rest of this
    // file uses for every other type crossing it.
    model->control.set_callback([callback, user_data](const nirnay::Progress& progress) {
      const nirnay_progress c_progress{static_cast<int>(progress.phase),
                                        static_cast<long>(progress.iterations),
                                        static_cast<long>(progress.nodes),
                                        static_cast<long>(progress.open_nodes),
                                        progress.objective,
                                        progress.best_bound,
                                        progress.gap,
                                        progress.elapsed_seconds};
      return callback(&c_progress, user_data) != 0;
    });
    return ok();
  });
}

nirnay_status nirnay_model_interrupt(nirnay_model* model) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  model->control.interrupt();
  return ok();
}

int nirnay_model_num_cols(const nirnay_model* model) {
  return model == nullptr ? 0 : static_cast<int>(model->model.col_cost.size());
}

int nirnay_model_num_rows(const nirnay_model* model) {
  return model == nullptr ? 0 : static_cast<int>(model->model.row_lower.size());
}

int nirnay_model_num_nonzeros(const nirnay_model* model) {
  if (model == nullptr) return 0;
  // Pending triplets when the caller built the model programmatically; the frozen matrix
  // when it was read from a file. Reporting only one of the two would make this function
  // answer a different question depending on how the handle was populated.
  const int pending = static_cast<int>(model->entries.size());
  return pending > 0 ? pending : static_cast<int>(model->model.matrix.num_nonzeros());
}

namespace {

/// Materialise the pending triplets into the Model's frozen matrices.
///
/// Called on a COPY at solve time rather than mutating the handle, so that a caller can
/// solve, add another column, and solve again without the first solve having frozen
/// anything underneath it.
void materialise(const nirnay_model& handle, nirnay::Model* out) {
  *out = handle.model;
  const auto rows = static_cast<nirnay::Index>(out->row_lower.size());
  const auto cols = static_cast<nirnay::Index>(out->col_cost.size());

  if (!handle.entries.empty() || out->matrix.num_nonzeros() == 0) {
    out->matrix.reset(rows, cols);
    out->matrix.reserve(handle.entries.size());
    for (const auto& entry : handle.entries) {
      out->matrix.add_entry(static_cast<nirnay::Index>(entry.first.first),
                            static_cast<nirnay::Index>(entry.first.second), entry.second);
    }
    out->matrix.finalize();
  }

  if (!handle.quadratic.empty()) {
    out->hessian.reset(cols, cols);
    out->hessian.reserve(handle.quadratic.size());
    for (const auto& entry : handle.quadratic) {
      out->hessian.add_entry(static_cast<nirnay::Index>(entry.first.first),
                             static_cast<nirnay::Index>(entry.first.second), entry.second);
    }
    out->hessian.finalize();
  }
}

}  // namespace

nirnay_status nirnay_model_validate(const nirnay_model* model) {
  if (model == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> nirnay_status {
    nirnay::Model built;
    materialise(*model, &built);
    const std::string problem = built.validate();
    if (!problem.empty()) return fail(NIRNAY_ERROR_MODEL, problem);
    return ok();
  });
}

// ---- Options -------------------------------------------------------------------------------

nirnay_options* nirnay_options_create(void) {
  try {
    return new nirnay_options();
  } catch (...) {
    return nullptr;
  }
}

void nirnay_options_free(nirnay_options* options) {
  delete options;
}

nirnay_status nirnay_options_set_bool(nirnay_options* options, const char* name, int value) {
  if (options == nullptr || name == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "options or name is null");
  }
  const nirnay_status check = check_option(name, nirnay::OptionType::Bool);
  if (check != NIRNAY_OK) return check;
  return guarded([&]() -> nirnay_status {
    options->options.set_bool(name, value != 0);
    return ok();
  });
}

nirnay_status nirnay_options_set_int(nirnay_options* options, const char* name, long value) {
  if (options == nullptr || name == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "options or name is null");
  }
  const nirnay_status check = check_option(name, nirnay::OptionType::Int);
  if (check != NIRNAY_OK) return check;
  return guarded([&]() -> nirnay_status {
    options->options.set_int(name, static_cast<std::int64_t>(value));
    return ok();
  });
}

nirnay_status nirnay_options_set_double(nirnay_options* options, const char* name,
                                          double value) {
  if (options == nullptr || name == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "options or name is null");
  }
  const nirnay_status check = check_option(name, nirnay::OptionType::Double);
  if (check != NIRNAY_OK) return check;
  return guarded([&]() -> nirnay_status {
    options->options.set_double(name, value);
    return ok();
  });
}

nirnay_status nirnay_options_set_string(nirnay_options* options, const char* name,
                                          const char* value) {
  if (options == nullptr || name == nullptr || value == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "options, name or value is null");
  }
  const nirnay_status check = check_option(name, nirnay::OptionType::String);
  if (check != NIRNAY_OK) return check;
  return guarded([&]() -> nirnay_status {
    options->options.set_string(name, value);
    return ok();
  });
}

// ---- Solve ---------------------------------------------------------------------------------

nirnay_status nirnay_solve(const nirnay_model* model, const nirnay_options* options,
                             nirnay_solution** solution) {
  if (model == nullptr || solution == nullptr) {
    return fail(NIRNAY_ERROR_ARGUMENT, "model or solution pointer is null");
  }
  *solution = nullptr;
  return guarded([&]() -> nirnay_status {
    nirnay::Model built;
    materialise(*model, &built);

    nirnay::Options effective;
    if (options != nullptr) effective = options->options;

    // Each solve starts with a clear interrupt flag and throttle history, so an
    // interrupt() called before or after a solve never reaches into an unrelated later one
    // (see nirnay_model_interrupt); the registered callback itself survives.
    model->control.reset();

    auto* result = new nirnay_solution();
    result->solution = nirnay::solve(built, effective, &model->control);
    *solution = result;
    return ok();
  });
}

void nirnay_solution_free(nirnay_solution* solution) {
  delete solution;
}

nirnay_solve_status nirnay_solution_status(const nirnay_solution* solution) {
  return solution == nullptr ? NIRNAY_NOT_SOLVED : to_c_status(solution->solution.status);
}

const char* nirnay_solution_message(const nirnay_solution* solution) {
  return solution == nullptr ? "" : solution->solution.message.c_str();
}

double nirnay_solution_objective(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.objective;
}

double nirnay_solution_dual_bound(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.dual_bound;
}

long nirnay_solution_iterations(const nirnay_solution* solution) {
  return solution == nullptr ? 0 : static_cast<long>(solution->solution.iterations);
}

long nirnay_solution_nodes(const nirnay_solution* solution) {
  return solution == nullptr ? 0 : static_cast<long>(solution->solution.nodes);
}

double nirnay_solution_seconds(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.solve_seconds;
}

double nirnay_solution_primal_infeasibility(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.primal_infeasibility;
}

double nirnay_solution_dual_infeasibility(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.dual_infeasibility;
}

double nirnay_solution_integrality_violation(const nirnay_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.integrality_violation;
}

nirnay_status nirnay_solution_col_values(const nirnay_solution* solution, double* values,
                                           int count) {
  if (solution == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.col_value, values, count, "column values");
}

nirnay_status nirnay_solution_row_activities(const nirnay_solution* solution, double* values,
                                               int count) {
  if (solution == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.row_activity, values, count, "row activities");
}

nirnay_status nirnay_solution_row_duals(const nirnay_solution* solution, double* values,
                                          int count) {
  if (solution == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.row_dual, values, count, "row duals");
}

nirnay_status nirnay_solution_col_duals(const nirnay_solution* solution, double* values,
                                          int count) {
  if (solution == nullptr) return fail(NIRNAY_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.col_dual, values, count, "column duals");
}

}  // extern "C"

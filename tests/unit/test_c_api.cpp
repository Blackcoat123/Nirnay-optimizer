// SPDX-License-Identifier: Apache-2.0
// NIRNAY - C API tests.
//
// These go through include/nirnay/nirnay.h ONLY. Nothing here includes model.hpp or calls
// nirnay::solve, because the thing under test is the boundary rather than the solver: a
// test that reached past the header would still pass if the C surface were wired to the
// wrong field, which is precisely the defect this layer can introduce and the core cannot.
//
// The answers are hand-derived rather than taken from a previous run, so a wrong wiring
// fails here instead of being frozen in as expected output.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nirnay/nirnay.h"

#include "support/temp_file.hpp"

namespace {

using nirnay::testing::TempFile;

/// RAII for the C handles, so a failing assertion cannot leak them.
struct ModelHandle {
  nirnay_model* handle = nirnay_model_create();
  ~ModelHandle() { nirnay_model_free(handle); }
  operator nirnay_model*() const { return handle; }
};

struct SolutionHandle {
  nirnay_solution* handle = nullptr;
  ~SolutionHandle() { nirnay_solution_free(handle); }
};

TEST(CApi, ReportsAVersionAndAnInfinity) {
  ASSERT_NE(nirnay_version(), nullptr);
  EXPECT_FALSE(std::string(nirnay_version()).empty());
  EXPECT_TRUE(std::isinf(nirnay_infinity()));
  EXPECT_GT(nirnay_infinity(), 0.0);
}

TEST(CApi, SolvesAnLpBuiltEntirelyThroughTheCSurface) {
  //   maximise  3x + 2y
  //   s.t.      x +  y <= 4
  //             x + 3y <= 6
  //             0 <= x <= 3,  y >= 0
  //
  // The vertex where both rows are tight is x = 3, y = 1, objective 11. x is at its upper
  // bound there, so this also exercises a boxed column rather than only the origin cone.
  ModelHandle model;
  ASSERT_NE(model.handle, nullptr);

  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, 3.0, 0.0, 3.0, 0, "x", &x), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, 2.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  EXPECT_EQ(x, 0);
  EXPECT_EQ(y, 1);

  int r0 = -1;
  int r1 = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 4.0, "c0", &r0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 6.0, "c1", &r1), NIRNAY_OK);

  ASSERT_EQ(nirnay_model_set_coefficient(model, r0, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, r0, y, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, r1, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, r1, y, 3.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_maximize(model, 1), NIRNAY_OK);

  EXPECT_EQ(nirnay_model_num_cols(model), 2);
  EXPECT_EQ(nirnay_model_num_rows(model), 2);
  EXPECT_EQ(nirnay_model_num_nonzeros(model), 4);
  ASSERT_EQ(nirnay_model_validate(model), NIRNAY_OK) << nirnay_last_error();

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  ASSERT_NE(solution.handle, nullptr);
  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL)
      << nirnay_solution_message(solution.handle);
  EXPECT_NEAR(nirnay_solution_objective(solution.handle), 11.0, 1e-9);

  std::vector<double> values(2, 0.0);
  ASSERT_EQ(nirnay_solution_col_values(solution.handle, values.data(), 2), NIRNAY_OK);
  EXPECT_NEAR(values[0], 3.0, 1e-9);
  EXPECT_NEAR(values[1], 1.0, 1e-9);

  std::vector<double> activities(2, 0.0);
  ASSERT_EQ(nirnay_solution_row_activities(solution.handle, activities.data(), 2), NIRNAY_OK);
  EXPECT_NEAR(activities[0], 4.0, 1e-9);
  EXPECT_NEAR(activities[1], 6.0, 1e-9);

  // The MEASURED quality, which is what a caller writing its own acceptance test should read.
  EXPECT_LE(nirnay_solution_primal_infeasibility(solution.handle), 1e-7);
  EXPECT_GT(nirnay_solution_iterations(solution.handle), 0);
}

TEST(CApi, SettingACoefficientTwiceReplacesItRatherThanSummingIt) {
  // The MPS reader treats a repeated entry as an error, because in a FILE it is a defect.
  // Through an API, overwriting a cell is ordinary, and summing would silently double a
  // coefficient - a change the caller cannot see in the answer. The two layers therefore
  // differ deliberately, and that difference is worth pinning.
  //
  //   minimise x  s.t.  2x >= 6,  x >= 0   ->   x = 3.
  // If the second set_coefficient summed onto the first, the row would read 3x >= 6 and the
  // answer would be 2.
  ModelHandle model;
  int x = -1;
  int row = -1;
  ASSERT_EQ(nirnay_model_add_column(model, 1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_row(model, 6.0, nirnay_infinity(), "c", &row), NIRNAY_OK);

  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 2.0), NIRNAY_OK);
  EXPECT_EQ(nirnay_model_num_nonzeros(model), 1) << "the entry was duplicated, not replaced";

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK);
  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL);
  EXPECT_NEAR(nirnay_solution_objective(solution.handle), 3.0, 1e-9);

  // Zero removes the entry, leaving an empty row rather than a zero-valued one.
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 0.0), NIRNAY_OK);
  EXPECT_EQ(nirnay_model_num_nonzeros(model), 0);
}

TEST(CApi, SolvesAQpAndTakesTheHessianInEitherTriangle) {
  // minimise 0.5 * (2x^2 + 2y^2) - 2x - 6y  subject to x + y <= 3, x, y >= 0.
  // Unconstrained stationary point is (1, 3), which violates the row, so the optimum sits on
  // x + y = 3. Substituting y = 3 - x: f(x) = x^2 + (3-x)^2 - 2x - 6(3-x) = 2x^2 - 2x - 9,
  // minimised at x = 0.5, y = 2.5, objective 2(0.25) - 1 - 9 = -9.5.
  ModelHandle model;
  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -2.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, -6.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 3.0, "c", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, y, 1.0), NIRNAY_OK);

  // Diagonal entries, and note the objective carries the 0.5 - so Q_xx = 2 means x^2.
  ASSERT_EQ(nirnay_model_set_quadratic_coefficient(model, x, x, 2.0), NIRNAY_OK);
  // Deliberately the UPPER index pair for the second one. Q is symmetric and only the lower
  // triangle is stored, so (y, y) is the same cell either way; the ordering matters for
  // off-diagonals and is asserted below.
  ASSERT_EQ(nirnay_model_set_quadratic_coefficient(model, y, y, 2.0), NIRNAY_OK);

  nirnay_options* options = nirnay_options_create();
  ASSERT_NE(options, nullptr);
  ASSERT_EQ(nirnay_options_set_bool(options, "log_to_console", 0), NIRNAY_OK);
  ASSERT_EQ(nirnay_options_set_double(options, "qp_tolerance", 1e-11), NIRNAY_OK);
  ASSERT_EQ(nirnay_options_set_int(options, "iteration_limit", 500000), NIRNAY_OK);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, options, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  nirnay_options_free(options);

  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL)
      << nirnay_solution_message(solution.handle);
  EXPECT_NEAR(nirnay_solution_objective(solution.handle), -9.5, 1e-5);

  std::vector<double> values(2, 0.0);
  ASSERT_EQ(nirnay_solution_col_values(solution.handle, values.data(), 2), NIRNAY_OK);
  EXPECT_NEAR(values[0], 0.5, 1e-4);
  EXPECT_NEAR(values[1], 2.5, 1e-4);
}

TEST(CApi, AnOffDiagonalHessianEntryMeansTheSameThingInEitherOrder) {
  // (i, j) and (j, i) name ONE entry of a symmetric Q. A wrapper that stored them separately
  // would set the coefficient twice and, on the second call, either double it or overwrite a
  // different cell. Building the same problem both ways and comparing the objective pins the
  // meaning rather than the storage.
  const auto build = [](bool upper_first) {
    nirnay_model* model = nirnay_model_create();
    nirnay_model_add_column(model, 0.0, -10.0, 10.0, 0, "a", nullptr);
    nirnay_model_add_column(model, 0.0, -10.0, 10.0, 0, "b", nullptr);
    int row = -1;
    nirnay_model_add_row(model, 2.0, 2.0, "c", &row);
    nirnay_model_set_coefficient(model, row, 0, 1.0);
    nirnay_model_set_coefficient(model, row, 1, 1.0);
    nirnay_model_set_quadratic_coefficient(model, 0, 0, 2.0);
    nirnay_model_set_quadratic_coefficient(model, 1, 1, 2.0);
    if (upper_first) {
      nirnay_model_set_quadratic_coefficient(model, 0, 1, 1.0);
    } else {
      nirnay_model_set_quadratic_coefficient(model, 1, 0, 1.0);
    }
    return model;
  };

  double objectives[2] = {0.0, 0.0};
  for (int variant = 0; variant < 2; ++variant) {
    nirnay_model* model = build(variant == 0);
    nirnay_options* options = nirnay_options_create();
    nirnay_options_set_bool(options, "log_to_console", 0);
    nirnay_options_set_double(options, "qp_tolerance", 1e-11);
    nirnay_options_set_int(options, "iteration_limit", 500000);

    nirnay_solution* solution = nullptr;
    ASSERT_EQ(nirnay_solve(model, options, &solution), NIRNAY_OK) << nirnay_last_error();
    ASSERT_EQ(nirnay_solution_status(solution), NIRNAY_OPTIMAL)
        << nirnay_solution_message(solution);
    objectives[variant] = nirnay_solution_objective(solution);

    nirnay_solution_free(solution);
    nirnay_options_free(options);
    nirnay_model_free(model);
  }
  EXPECT_NEAR(objectives[0], objectives[1], 1e-6)
      << "the index order changed the problem, so the two triangles are being stored apart";
}

TEST(CApi, ReadsAModelFromAFile) {
  const TempFile file(
      "NAME          TINY\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1           4.0\n"
      "ENDATA\n",
      ".mps");

  ModelHandle model;
  ASSERT_EQ(nirnay_model_read(model, file.path().c_str()), NIRNAY_OK) << nirnay_last_error();
  EXPECT_EQ(nirnay_model_num_cols(model), 1);
  EXPECT_EQ(nirnay_model_num_rows(model), 1);
  EXPECT_EQ(nirnay_model_num_nonzeros(model), 1);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK);
  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL);
  EXPECT_NEAR(nirnay_solution_objective(solution.handle), 4.0, 1e-9);
}

TEST(CApi, AFailedReadLeavesTheHandleUntouched) {
  // The worst outcome for a failed read is a HALF-populated handle: the caller sees an error
  // and still holds something that solves, answering a question no one asked.
  ModelHandle model;
  ASSERT_EQ(nirnay_model_add_column(model, 1.0, 0.0, 1.0, 0, "keep", nullptr), NIRNAY_OK);

  const TempFile broken("NAME          BAD\nROWS\n N  COST\nCOLUMNS\n    X  NOSUCHROW  1.0\n",
                        ".mps");
  EXPECT_EQ(nirnay_model_read(model, broken.path().c_str()), NIRNAY_ERROR_IO);
  EXPECT_FALSE(std::string(nirnay_last_error()).empty()) << "a failure with no explanation";

  EXPECT_EQ(nirnay_model_num_cols(model), 1) << "the failed read modified the handle";
}

TEST(CApi, RejectsBadArgumentsRatherThanCrashing) {
  // A C caller gets null and out-of-range wrong eventually. Each of these would be undefined
  // behaviour if the boundary did not check, so the checks are the feature.
  EXPECT_EQ(nirnay_model_set_maximize(nullptr, 1), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_model_read(nullptr, "x.mps"), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_options_set_bool(nullptr, "presolve", 0), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_model_num_cols(nullptr), 0);
  EXPECT_EQ(nirnay_solution_status(nullptr), NIRNAY_NOT_SOLVED);

  ModelHandle model;
  nirnay_model_add_column(model, 1.0, 0.0, 1.0, 0, "x", nullptr);
  nirnay_model_add_row(model, 0.0, 1.0, "r", nullptr);
  EXPECT_EQ(nirnay_model_set_coefficient(model, 5, 0, 1.0), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_model_set_coefficient(model, 0, 5, 1.0), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_model_set_coefficient(model, -1, 0, 1.0), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_model_set_quadratic_coefficient(model, 0, 9, 1.0), NIRNAY_ERROR_ARGUMENT);

  // A wrong buffer size must be refused outright rather than partially filled: a caller with
  // the dimension wrong is about to misread every number it copies.
  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK);
  double one = 0.0;
  EXPECT_EQ(nirnay_solution_col_values(solution.handle, &one, 7), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_solution_col_values(solution.handle, nullptr, 1), NIRNAY_ERROR_ARGUMENT);
}

TEST(CApi, RejectsAnUnknownOption) {
  nirnay_options* options = nirnay_options_create();
  ASSERT_NE(options, nullptr);
  // The registry refuses unknown names rather than storing them, so a typo in a caller's
  // option string fails where it is written instead of being silently ignored for a whole run.
  const nirnay_status status = nirnay_options_set_double(options, "no_such_option", 1.0);
  EXPECT_NE(status, NIRNAY_OK);
  EXPECT_FALSE(std::string(nirnay_last_error()).empty());
  nirnay_options_free(options);
}

TEST(CApi, SolvesAMilpAndReportsIntegrality) {
  //   maximise x + y   s.t.  2x + 2y <= 3,  x, y in {0, 1}
  // The relaxation gives x = y = 0.75 for 1.5; the integer optimum is any single unit, 1.
  ModelHandle model;
  ASSERT_EQ(nirnay_model_add_column(model, 1.0, 0.0, 1.0, 1, "x", nullptr), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, 1.0, 0.0, 1.0, 1, "y", nullptr), NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 3.0, "c", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, 0, 2.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, 1, 2.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_maximize(model, 1), NIRNAY_OK);

  nirnay_options* options = nirnay_options_create();
  nirnay_options_set_bool(options, "log_to_console", 0);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, options, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  nirnay_options_free(options);

  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL)
      << nirnay_solution_message(solution.handle);
  EXPECT_NEAR(nirnay_solution_objective(solution.handle), 1.0, 1e-9);
  EXPECT_LE(nirnay_solution_integrality_violation(solution.handle), 1e-6);

  std::vector<double> values(2, 0.0);
  ASSERT_EQ(nirnay_solution_col_values(solution.handle, values.data(), 2), NIRNAY_OK);
  for (double v : values) {
    EXPECT_TRUE(std::fabs(v) < 1e-6 || std::fabs(v - 1.0) < 1e-6) << "fractional: " << v;
  }
}

TEST(CApi, AModelCanBeExtendedAndResolvedWithoutBeingFrozenByTheFirstSolve) {
  // materialise() builds the matrix on a COPY so the handle stays mutable. Without that, the
  // first solve would freeze the sparse matrix and the second add_column would either be
  // ignored or assert - the kind of failure that only appears in a caller doing something
  // perfectly reasonable.
  ModelHandle model;
  ASSERT_EQ(nirnay_model_add_column(model, 1.0, 0.0, nirnay_infinity(), 0, "x", nullptr),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, 2.0, nirnay_infinity(), "c", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, 0, 1.0), NIRNAY_OK);

  SolutionHandle first;
  ASSERT_EQ(nirnay_solve(model, nullptr, &first.handle), NIRNAY_OK);
  ASSERT_EQ(nirnay_solution_status(first.handle), NIRNAY_OPTIMAL);
  EXPECT_NEAR(nirnay_solution_objective(first.handle), 2.0, 1e-9);

  // A cheaper second column that can satisfy the same row.
  ASSERT_EQ(nirnay_model_add_column(model, 0.25, 0.0, nirnay_infinity(), 0, "y", nullptr),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, 1, 1.0), NIRNAY_OK);
  EXPECT_EQ(nirnay_model_num_cols(model), 2);

  SolutionHandle second;
  ASSERT_EQ(nirnay_solve(model, nullptr, &second.handle), NIRNAY_OK) << nirnay_last_error();
  ASSERT_EQ(nirnay_solution_status(second.handle), NIRNAY_OPTIMAL);
  EXPECT_NEAR(nirnay_solution_objective(second.handle), 0.5, 1e-9);
}

// =========================================================================================
// Progress callback and interruption (#223)
// =========================================================================================

TEST(CApi, ProgressCallbackReceivesSnapshotsAndUserData) {
  // min -x - y  s.t.  x + y <= 100, x, y >= 0. What matters is not the answer but that the
  // callback is reached at all, carries a sane snapshot, and gets its user_data back
  // unchanged - the three things a C caller cannot get any other way than by calling it.
  ModelHandle model;
  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 100.0, "cap", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, y, 1.0), NIRNAY_OK);

  struct UserData {
    int calls = 0;
    void* self = nullptr;  ///< set from inside the callback, to the pointer IT was given
  } user_data;

  const auto callback = [](const nirnay_progress* progress, void* raw) -> int {
    auto* data = static_cast<UserData*>(raw);
    ++data->calls;
    data->self = raw;
    EXPECT_GE(progress->elapsed_seconds, 0.0);
    return 0;  // do not stop
  };

  ASSERT_EQ(nirnay_set_callback(model, callback, &user_data), NIRNAY_OK);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL);
  EXPECT_GE(user_data.calls, 1);
  // user_data round-trips as the SAME pointer, not a copy passed through by value somewhere
  // in the C++/C boundary - a wrapper bug that swapped in a different pointer would fail
  // this even though the calls above still "worked" by writing through the wrong address.
  EXPECT_EQ(user_data.self, &user_data);
}

TEST(CApi, ANonZeroCallbackReturnStopsTheSolveWithInterrupted) {
  // Two columns, not one: a single-column "x <= 100" model is exactly the shape presolve
  // fixes outright, reducing it to a 0x0 LP that the simplex solves without ever running a
  // loop iteration - and so without ever calling poll(). Two columns sharing one row is not
  // individually fixable, so the model reaches the real engine.
  ModelHandle model;
  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 100.0, "cap", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, y, 1.0), NIRNAY_OK);

  const auto stop_immediately = [](const nirnay_progress*, void*) -> int { return 1; };
  ASSERT_EQ(nirnay_set_callback(model, stop_immediately, nullptr), NIRNAY_OK);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  EXPECT_EQ(nirnay_solution_status(solution.handle), NIRNAY_INTERRUPTED);

  std::vector<double> values(2, -1.0);
  EXPECT_EQ(nirnay_solution_col_values(solution.handle, values.data(), 2), NIRNAY_OK)
      << "kInterrupted claims a point, same as a limit does";
}

TEST(CApi, ClearingTheCallbackWithNullStopsItBeingCalled) {
  ModelHandle model;
  int x = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 100.0, "cap", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);

  int calls = 0;
  const auto counting = [](const nirnay_progress*, void* raw) -> int {
    ++*static_cast<int*>(raw);
    return 0;
  };
  ASSERT_EQ(nirnay_set_callback(model, counting, &calls), NIRNAY_OK);
  ASSERT_EQ(nirnay_set_callback(model, nullptr, nullptr), NIRNAY_OK);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  ASSERT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL);
  EXPECT_EQ(calls, 0);
}

TEST(CApi, PreSolveInterruptHasNoEffectOnThatSolve) {
  // nirnay_solve() clears the interrupt flag before it does anything else (see
  // nirnay_solve's own comment: "each solve starts with a clear flag ... an interrupt()
  // called before or after a solve has no effect on the next one"). This is what keeps a
  // handle reused for a LATER, unrelated solve from being silently poisoned by an interrupt
  // nobody ever cleared - the real, intended use is interrupt() called WHILE a solve is
  // running (from another thread; see the Python bindings' cross-thread test, which drives
  // this exact entry point), not before it starts.
  ModelHandle model;
  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 100.0, "cap", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, y, 1.0), NIRNAY_OK);

  ASSERT_EQ(nirnay_model_interrupt(model), NIRNAY_OK);

  SolutionHandle solution;
  ASSERT_EQ(nirnay_solve(model, nullptr, &solution.handle), NIRNAY_OK)
      << nirnay_last_error();
  EXPECT_EQ(nirnay_solution_status(solution.handle), NIRNAY_OPTIMAL);
}

TEST(CApi, AnInterruptedSolveDoesNotPoisonTheNextOne) {
  // The callback is what actually interrupts solve 1 here (deterministic, no race) - once,
  // via a flag in user_data, so it does not also interrupt solve 2. What is under test is
  // that nirnay_solve's own reset() genuinely clears SolveControl's state between calls:
  // solve 2 is unaffected by solve 1 having been interrupted.
  ModelHandle model;
  int x = -1;
  int y = -1;
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "x", &x),
            NIRNAY_OK);
  ASSERT_EQ(nirnay_model_add_column(model, -1.0, 0.0, nirnay_infinity(), 0, "y", &y),
            NIRNAY_OK);
  int row = -1;
  ASSERT_EQ(nirnay_model_add_row(model, -nirnay_infinity(), 100.0, "cap", &row), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, x, 1.0), NIRNAY_OK);
  ASSERT_EQ(nirnay_model_set_coefficient(model, row, y, 1.0), NIRNAY_OK);

  bool already_stopped_once = false;
  const auto stop_only_the_first_time = [](const nirnay_progress*, void* raw) -> int {
    auto* flag = static_cast<bool*>(raw);
    if (*flag) return 0;
    *flag = true;
    return 1;
  };
  ASSERT_EQ(nirnay_set_callback(model, stop_only_the_first_time, &already_stopped_once),
            NIRNAY_OK);

  SolutionHandle first;
  ASSERT_EQ(nirnay_solve(model, nullptr, &first.handle), NIRNAY_OK) << nirnay_last_error();
  ASSERT_EQ(nirnay_solution_status(first.handle), NIRNAY_INTERRUPTED);

  SolutionHandle second;
  ASSERT_EQ(nirnay_solve(model, nullptr, &second.handle), NIRNAY_OK) << nirnay_last_error();
  EXPECT_EQ(nirnay_solution_status(second.handle), NIRNAY_OPTIMAL);
}

TEST(CApi, InterruptFromNullModelIsAnArgumentError) {
  EXPECT_EQ(nirnay_model_interrupt(nullptr), NIRNAY_ERROR_ARGUMENT);
  EXPECT_EQ(nirnay_set_callback(nullptr, nullptr, nullptr), NIRNAY_ERROR_ARGUMENT);
}

}  // namespace

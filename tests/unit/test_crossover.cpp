// SPDX-License-Identifier: Apache-2.0
// NIRNAY - crossover from a first-order or interior-point answer to an optimal basis.
//
// The property that matters is the contract in src/simplex/crossover.hpp: a crossover either
// hands back the simplex's optimal vertex - the same objective the simplex reports, a real
// basis, the pivots accounted for - or leaves the first answer exactly as it was.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "nirnay/io.hpp"
#include "nirnay/logging.hpp"
#include "nirnay/model.hpp"
#include "nirnay/timer.hpp"
#include "nirnay/options.hpp"
#include "oracles/lp_generator.hpp"
#include "simplex/crossover.hpp"

namespace nirnay {
namespace {

std::string repo_path(const std::string& relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

Model read_netlib(const std::string& name) {
  Model model;
  const io::ReadResult read = io::read_model(repo_path("data/netlib/" + name + ".mps"), &model);
  EXPECT_TRUE(read.ok) << read.error;
  return model;
}

Options with(const char* algorithm, const char* crossover_mode) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", algorithm);
  options.set_string("crossover", crossover_mode);
  options.set_double("pdhg_tolerance", 1e-8);
  return options;
}

bool has_basis(const Solution& s) {
  return std::count(s.col_status.begin(), s.col_status.end(), BasisStatus::kBasic) +
             std::count(s.row_status.begin(), s.row_status.end(), BasisStatus::kBasic) >
         0;
}

TEST(Crossover, IdentifiesExactlyOneBasicVariablePerRow) {
  std::mt19937_64 rng(20260929);
  oracle::GeneratorConfig config;
  config.max_rows = 8;
  config.max_cols = 8;
  for (int trial = 0; trial < 40; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution first = solve(model, with("pdhg", "off"));
    if (first.status != SolveStatus::kOptimal) continue;
    const WarmStart warm = identify_basis(model, first);
    const auto basic =
        std::count(warm.col_status.begin(), warm.col_status.end(), BasisStatus::kBasic) +
        std::count(warm.row_status.begin(), warm.row_status.end(), BasisStatus::kBasic);
    EXPECT_EQ(basic, model.num_rows());
  }
}

TEST(Crossover, PdhgFinishedByCrossoverIsTheSimplexVertex) {
  std::mt19937_64 rng(20260930);
  oracle::GeneratorConfig config;
  config.max_rows = 7;
  config.max_cols = 7;
  int compared = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution simplex = solve(model, with("dual-simplex", "off"));
    if (simplex.status != SolveStatus::kOptimal) continue;
    const Solution crossed = solve(model, with("pdhg", "on"));
    if (crossed.status != SolveStatus::kOptimal) continue;
    ++compared;
    const double scale = std::max(1.0, std::fabs(simplex.objective));
    EXPECT_LE(std::fabs(crossed.objective - simplex.objective) / scale, 1e-9)
        << instance.lp.to_text();
    if (crossed.algorithm.find("+crossover") != std::string::npos) {
      EXPECT_TRUE(has_basis(crossed)) << crossed.message;
      EXPECT_GE(crossed.iterations, crossed.crossover_iterations);
    }
  }
  EXPECT_GT(compared, 40) << "too few instances reached the comparison";
}

TEST(Crossover, FinishesTheInteriorPointOnNetlibToTheSimplexObjective) {
  for (const char* name : {"afiro", "sc50a", "blend", "share2b"}) {
    const Model model = read_netlib(name);
    const Solution simplex = solve(model, with("dual-simplex", "off"));
    const Solution crossed = solve(model, with("ipm", "on"));
    ASSERT_EQ(crossed.status, SolveStatus::kOptimal) << name << ": " << crossed.message;
    EXPECT_NE(crossed.algorithm.find("+crossover"), std::string::npos) << name;
    EXPECT_TRUE(has_basis(crossed)) << name;
    EXPECT_NEAR(crossed.objective, simplex.objective,
                1e-9 * std::max(1.0, std::fabs(simplex.objective)))
        << name;
  }
}

TEST(Crossover, OffLeavesTheFirstOrderAnswerAlone) {
  const Model model = read_netlib("afiro");
  // Presolve off: postsolve gives the rows and columns it removed statuses of their own,
  // and the question here is only what the first-order engine itself hands back.
  Options options = with("pdhg", "off");
  options.set_bool("presolve", false);
  const Solution s = solve(model, options);
  EXPECT_EQ(s.algorithm.find("+crossover"), std::string::npos) << s.algorithm;
  EXPECT_EQ(s.crossover_iterations, 0);
  EXPECT_FALSE(has_basis(s));
}

TEST(Crossover, AutoRunsOnASmallModel) {
  const Model model = read_netlib("sc105");
  const Solution s = solve(model, with("pdhg", "auto"));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.algorithm.find("+crossover"), std::string::npos) << s.algorithm;
}

TEST(Crossover, WithNoTimeTheFirstAnswerStandsUnchanged) {
  const Model model = read_netlib("adlittle");
  Options options = with("pdhg", "off");
  const Solution plain = solve(model, options);
  ASSERT_EQ(plain.status, SolveStatus::kOptimal) << plain.message;

  Solution copy = plain;
  Options no_time = with("pdhg", "on");
  no_time.set_double("crossover_max_seconds", 0.0);
  Logger quiet(nullptr);
  const Timer timer;
  crossover(&copy, model, no_time, quiet, timer, nullptr);
  EXPECT_EQ(copy.algorithm, plain.algorithm);
  EXPECT_DOUBLE_EQ(copy.objective, plain.objective);
  EXPECT_EQ(copy.status, plain.status);
}

}  // namespace
}  // namespace nirnay

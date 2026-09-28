// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the two PDHG schemes and the two backends, pinned to each other and to the simplex.
//
// What is promised (src/pdhg/backend.hpp):
//   * the CPU backend's answer does not depend on the thread count, to the bit;
//   * the adaptive and the Halpern scheme both reach the simplex's optimum;
//   * the GPU backend runs the same algorithm - on a machine with a CUDA device and a build
//     with the backend, its answers agree with the CPU's; elsewhere those tests are skipped,
//     and say so, rather than passing vacuously.

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "nirnay/io.hpp"
#include "nirnay/model.hpp"
#include "nirnay/options.hpp"
#include "nirnay/version.hpp"
#include "oracles/lp_generator.hpp"

namespace nirnay {
namespace {

std::string repo_path(const std::string& relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

Options first_order(const char* method) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_string("pdhg_method", method);
  options.set_double("pdhg_tolerance", 1e-8);
  options.set_bool("pdhg_polish", false);
  options.set_string("crossover", "off");
  return options;
}

Model netlib(const std::string& name) {
  Model model;
  const io::ReadResult read = io::read_model(repo_path("data/netlib/" + name + ".mps"), &model);
  EXPECT_TRUE(read.ok) << read.error;
  return model;
}

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

TEST(PdhgBackends, TheCpuAnswerDoesNotDependOnTheThreadCount) {
  // israel is the largest committed instance; share2b needs the most iterations. Neither is
  // large enough for the loops to fork unless the thresholds are crossed, so the thresholds
  // are exercised by the scale family in bench/, and the chunked reductions here.
  for (const char* name : {"israel", "share2b"}) {
    const Model model = netlib(name);
    Options one = first_order("halpern");
    one.set_int("threads", 1);
    Options many = first_order("halpern");
    many.set_int("threads", 8);
    const Solution a = solve(model, one);
    const Solution b = solve(model, many);
    ASSERT_EQ(a.status, SolveStatus::kOptimal) << name << ": " << a.message;
    EXPECT_EQ(a.status, b.status) << name;
    EXPECT_EQ(a.iterations, b.iterations) << name;
    EXPECT_TRUE(same_bits(a.objective, b.objective))
        << name << ": " << a.objective << " vs " << b.objective;
  }
}

TEST(PdhgBackends, BothSchemesReachTheSimplexOptimum) {
  std::mt19937_64 rng(20260928);
  oracle::GeneratorConfig config;
  config.max_rows = 6;
  config.max_cols = 6;
  Options simplex;
  simplex.set_bool("log_to_console", false);
  simplex.set_string("algorithm", "dual-simplex");
  int compared = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution exact = solve(model, simplex);
    if (exact.status != SolveStatus::kOptimal) continue;
    for (const char* method : {"adaptive", "halpern"}) {
      const Solution s = solve(model, first_order(method));
      if (s.status != SolveStatus::kOptimal) continue;
      ++compared;
      const double scale = std::max(1.0, std::fabs(exact.objective));
      EXPECT_LE(std::fabs(s.objective - exact.objective) / scale, 1e-6)
          << method << "\n"
          << instance.lp.to_text();
    }
  }
  EXPECT_GT(compared, 80) << "too few instances converged for this to mean anything";
}

TEST(PdhgBackends, HalpernReachesOptimalOnEveryCommittedNetlibInstance) {
  // The measurement behind making it the default (docs/BENCHMARKS.md section 1e): all nine,
  // with neither the polish nor the crossover to finish the job.
  for (const char* name : {"afiro", "sc50a", "sc50b", "sc105", "adlittle", "blend", "israel",
                           "share2b", "stocfor1"}) {
    const Solution s = solve(netlib(name), first_order("halpern"));
    EXPECT_EQ(s.status, SolveStatus::kOptimal) << name << ": " << s.message;
  }
}

/// A GPU solve of `model`, or a skip when there is no GPU to run it on.
bool gpu_solve(const Model& model, Solution* out) {
  Options options = first_order("halpern");
  options.set_bool("gpu", true);
  *out = solve(model, options);
  return out->algorithm == "pdhg-gpu";
}

TEST(PdhgBackends, TheGpuAgreesWithTheCpu) {
  if (!cuda_enabled()) GTEST_SKIP() << "this build has no CUDA backend";
  for (const char* name : {"afiro", "sc105", "blend", "adlittle", "israel", "stocfor1"}) {
    const Model model = netlib(name);
    Solution gpu;
    if (!gpu_solve(model, &gpu)) GTEST_SKIP() << "no CUDA device answered: " << gpu.message;
    const Solution cpu = solve(model, first_order("halpern"));
    ASSERT_EQ(gpu.status, SolveStatus::kOptimal) << name << ": " << gpu.message;
    ASSERT_EQ(cpu.status, SolveStatus::kOptimal) << name << ": " << cpu.message;
    const double scale = std::max(1.0, std::fabs(cpu.objective));
    EXPECT_LE(std::fabs(gpu.objective - cpu.objective) / scale, 1e-8) << name;
  }
}

TEST(PdhgBackends, TheGpuAgreesWithTheSimplexOnGeneratedInstances) {
  if (!cuda_enabled()) GTEST_SKIP() << "this build has no CUDA backend";
  std::mt19937_64 rng(20260927);
  oracle::GeneratorConfig config;
  config.max_rows = 6;
  config.max_cols = 6;
  Options simplex;
  simplex.set_bool("log_to_console", false);
  simplex.set_string("algorithm", "dual-simplex");
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution exact = solve(model, simplex);
    if (exact.status != SolveStatus::kOptimal) continue;
    Solution gpu;
    if (!gpu_solve(model, &gpu)) GTEST_SKIP() << "no CUDA device answered: " << gpu.message;
    if (gpu.status != SolveStatus::kOptimal) continue;
    ++compared;
    const double scale = std::max(1.0, std::fabs(exact.objective));
    EXPECT_LE(std::fabs(gpu.objective - exact.objective) / scale, 1e-6)
        << instance.lp.to_text();
  }
  EXPECT_GT(compared, 20);
}

}  // namespace
}  // namespace nirnay

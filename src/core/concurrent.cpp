// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the concurrent LP optimizer. See concurrent.hpp.

#include "core/concurrent.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "nirnay/timer.hpp"

namespace nirnay {
namespace {

/// How good an answer is, for choosing among racers none of which proved optimality.
int status_rank(SolveStatus status) {
  switch (status) {
    case SolveStatus::kOptimal: return 0;
    case SolveStatus::kInfeasible:
    case SolveStatus::kUnbounded: return 1;  // a proof, of a different verdict
    case SolveStatus::kFeasible: return 2;
    case SolveStatus::kIterationLimit:
    case SolveStatus::kTimeLimit:
    case SolveStatus::kNodeLimit: return 3;
    case SolveStatus::kInterrupted: return 4;
    default: return 5;
  }
}

/// A verdict that ends the race: an optimal point, or a certified infeasible/unbounded.
bool decisive(const Solution& s) {
  return s.status == SolveStatus::kOptimal ||
         (s.status == SolveStatus::kInfeasible && !s.farkas_dual.empty()) ||
         (s.status == SolveStatus::kUnbounded && !s.primal_ray.empty());
}

struct Lane {
  Solution result;
  SolveControl control;
  bool done = false;
  double finished_at = 0.0;
  int finish_order = -1;
};

}  // namespace

Solution race(const std::vector<Racer>& racers, Logger& logger, SolveControl* outer) {
  const Timer timer;
  std::vector<std::unique_ptr<Lane>> lanes;
  lanes.reserve(racers.size());
  for (std::size_t r = 0; r < racers.size(); ++r) lanes.push_back(std::make_unique<Lane>());

  std::mutex mutex;
  std::condition_variable changed;
  int finished = 0;

  logger.info("Concurrent: racing {} engines", racers.size());
  std::vector<std::thread> threads;
  threads.reserve(racers.size());
  for (std::size_t r = 0; r < racers.size(); ++r) {
    threads.emplace_back([&, r] {
      Logger silent(nullptr);
      Solution result;
      try {
        result = racers[r].run(silent, &lanes[r]->control);
      } catch (const std::exception& error) {
        result.status = SolveStatus::kNumericalError;
        result.message = fmt::format("the {} engine failed: {}", racers[r].name, error.what());
      }
      const std::lock_guard<std::mutex> lock(mutex);
      lanes[r]->result = std::move(result);
      lanes[r]->done = true;
      lanes[r]->finished_at = timer.elapsed_seconds();
      lanes[r]->finish_order = finished++;
      changed.notify_all();
    });
  }

  // Wait for a decisive answer, for everyone, or for the caller to interrupt.
  int winner = -1;
  {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
      for (std::size_t r = 0; r < lanes.size() && winner < 0; ++r) {
        if (lanes[r]->done && decisive(lanes[r]->result)) winner = static_cast<int>(r);
      }
      if (winner >= 0 || finished == static_cast<int>(lanes.size())) break;
      if (outer != nullptr && outer->is_interrupted()) break;
      changed.wait_for(lock, std::chrono::milliseconds(20));
    }
  }
  for (auto& lane : lanes) lane->control.interrupt();
  for (std::thread& thread : threads) thread.join();

  if (winner < 0) {
    // Nobody proved anything: keep the best answer by status, then by measured quality.
    for (std::size_t r = 0; r < lanes.size(); ++r) {
      const Solution& s = lanes[r]->result;
      if (winner < 0) {
        winner = static_cast<int>(r);
        continue;
      }
      const Solution& best = lanes[static_cast<std::size_t>(winner)]->result;
      const int a = status_rank(s.status);
      const int b = status_rank(best.status);
      const double sa = std::max(s.primal_infeasibility_scaled, s.dual_infeasibility_scaled);
      const double sb =
          std::max(best.primal_infeasibility_scaled, best.dual_infeasibility_scaled);
      if (a < b || (a == b && claims_a_point(s.status) && sa < sb)) winner = static_cast<int>(r);
    }
  }

  std::string others;
  for (std::size_t r = 0; r < lanes.size(); ++r) {
    if (static_cast<int>(r) == winner) continue;
    const Solution& s = lanes[r]->result;
    others += fmt::format("{}{} {} after {} iterations", others.empty() ? "" : ", ",
                          racers[r].name, to_string(s.status), s.iterations);
  }
  const auto w = static_cast<std::size_t>(winner);
  logger.info("Concurrent: {} won with {} at {:.3f}s; {}", racers[w].name,
              to_string(lanes[w]->result.status), lanes[w]->finished_at, others);
  Solution result = std::move(lanes[w]->result);
  result.message += fmt::format("{}won the concurrent race ({})",
                                result.message.empty() ? "" : "; ", others);
  result.algorithm = "concurrent:" + result.algorithm;
  return result;
}

}  // namespace nirnay

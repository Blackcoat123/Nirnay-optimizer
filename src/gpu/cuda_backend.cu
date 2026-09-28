// SPDX-License-Identifier: Apache-2.0
// NIRNAY - the CUDA backend for restarted PDHG (#16-#19). See src/pdhg/backend.hpp for the
// contract and src/pdhg/pdhg.cpp for the algorithm and its references.
//
// WHAT RUNS WHERE. Every vector of the iteration - the iterate, its PDHG output, the anchor,
// the best point, the running average, and each one's two matrix products - lives on the
// device for the whole solve. The two sparse products per iteration go through cuSPARSE
// (NVIDIA's vendor sparse BLAS, not a solver; docs/PROVENANCE.md) over explicit row-
// compressed copies of Ahat and Ahat', so both are row-parallel gathers. Everything else is
// a handful of fused element-wise kernels written here. A block of Halpern iterations is
// launched back to back with no host synchronisation at all; the host reads back three
// scalars at the end of the block, and whole vectors only when it asks for a point.
//
// DOUBLE PRECISION THROUGHOUT. No float anywhere, the project's rule for the numerical core
// (tolerances.hpp). The iteration is memory-bound, so FP64 costs bandwidth, not arithmetic.
//
// DETERMINISM. Reductions are two-pass: each block reduces a grid-strided slice in a fixed
// tree, then one block folds the per-block partials in block order. The grid depends only on
// the vector length and the device's SM count, so a run on a given device is reproducible
// bit for bit. The sparse products use CUSPARSE_SPMV_CSR_ALG2, cuSPARSE's deterministic CSR
// algorithm. Runs on different devices, or against the CPU backend, agree to rounding, and
// the final verdict is measured on the host either way.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "pdhg/backend.hpp"

namespace nirnay::pdhg {
namespace {

constexpr int kThreads = 256;
/// Partial-sum slots per reduction: enough for the widest kernel (four quantities).
constexpr int kMaxQuantities = 4;

// ---- Device helpers ------------------------------------------------------------------------

__device__ __forceinline__ bool finite_bound(double v) {
  return !(isinf(v) || fabs(v) >= 1e30);  // types.hpp: is_finite_bound, kMpsInfinity
}

__device__ __forceinline__ double project(double value, double lower, double upper) {
  if (finite_bound(lower) && value < lower) return lower;
  if (finite_bound(upper) && value > upper) return upper;
  return value;
}

__device__ __forceinline__ double dual_prox(double v, double sigma, double lower, double upper) {
  return v - sigma * project(v / sigma, lower, upper);
}

/// Block-wide sum in a fixed tree; every thread must call it.
__device__ double block_sum(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const double total = shared[0];
  __syncthreads();
  return total;
}

/// Block-wide max. fmax returns the other operand when one is NaN, which is the host's rule
/// for a complementarity product that is not a number: it is skipped.
__device__ double block_max(double value, double* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmax(shared[threadIdx.x],
                                                         shared[threadIdx.x + stride]);
    __syncthreads();
  }
  const double total = shared[0];
  __syncthreads();
  return total;
}

#define NIRNAY_GRID_STRIDE(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ---- The iteration --------------------------------------------------------------------------

/// T(z), primal half: xt = proj_X(x - tau (c + A'y)); optionally sum |xt - x|^2.
template <bool kMeasure>
__global__ void primal_output(int n, const double* __restrict__ x,
                              const double* __restrict__ aty, const double* __restrict__ c,
                              const double* __restrict__ l, const double* __restrict__ u,
                              double tau, double* __restrict__ xt, double* partials) {
  __shared__ double shared[kThreads];
  double dx = 0.0;
  NIRNAY_GRID_STRIDE(j, n) {
    const double next = project(x[j] - tau * (c[j] + aty[j]), l[j], u[j]);
    xt[j] = next;
    if (kMeasure) {
      const double d = next - x[j];
      dx += d * d;
    }
  }
  if (kMeasure) {
    const double total = block_sum(dx, shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = total;
  }
}

/// T(z), dual half, for an adaptive trial: yt = prox(y + sigma (2 A xt - A x)), with
/// |yt - y|^2 and (yt - y)'(A xt - A x).
__global__ void dual_trial(int m, const double* __restrict__ y, const double* __restrict__ ax,
                           const double* __restrict__ axt, const double* __restrict__ rl,
                           const double* __restrict__ ru, double sigma,
                           double* __restrict__ yt, double* partials) {
  __shared__ double shared[kThreads];
  double dy = 0.0;
  double cross = 0.0;
  NIRNAY_GRID_STRIDE(i, m) {
    const double v = y[i] + sigma * (2.0 * axt[i] - ax[i]);
    const double next = dual_prox(v, sigma, rl[i], ru[i]);
    yt[i] = next;
    const double d = next - y[i];
    dy += d * d;
    cross += d * (axt[i] - ax[i]);
  }
  const double dy_total = block_sum(dy, shared);
  const double cross_total = block_sum(cross, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = dy_total;
    partials[gridDim.x + blockIdx.x] = cross_total;
  }
}

/// T(z), dual half, then the Halpern update of y and - by linearity - of A x:
///   y  <- w ((1 + rho) yt  - rho y)  + (1 - w) y0
///   Ax <- w ((1 + rho) Axt - rho Ax) + (1 - w) Ax0
template <bool kMeasure>
__global__ void dual_halpern(int m, double* __restrict__ y, double* __restrict__ ax,
                             const double* __restrict__ axt, const double* __restrict__ rl,
                             const double* __restrict__ ru, double sigma, double w, double rho,
                             const double* __restrict__ y0, const double* __restrict__ ax0,
                             double* __restrict__ yt, double* partials) {
  __shared__ double shared[kThreads];
  double dy = 0.0;
  double cross = 0.0;
  NIRNAY_GRID_STRIDE(i, m) {
    const double axi = ax[i];
    const double axti = axt[i];
    const double yi = y[i];
    const double next = dual_prox(yi + sigma * (2.0 * axti - axi), sigma, rl[i], ru[i]);
    yt[i] = next;
    if (kMeasure) {
      const double d = yi - next;
      dy += d * d;
      cross += d * (axi - axti);
    }
    y[i] = w * ((1.0 + rho) * next - rho * yi) + (1.0 - w) * y0[i];
    ax[i] = w * ((1.0 + rho) * axti - rho * axi) + (1.0 - w) * ax0[i];
  }
  if (kMeasure) {
    const double dy_total = block_sum(dy, shared);
    const double cross_total = block_sum(cross, shared);
    if (threadIdx.x == 0) {
      partials[blockIdx.x] = dy_total;
      partials[gridDim.x + blockIdx.x] = cross_total;
    }
  }
}

/// The Halpern update of x and, by linearity, of A'y.
__global__ void primal_halpern(int n, double* __restrict__ x, double* __restrict__ aty,
                               const double* __restrict__ xt, const double* __restrict__ atyt,
                               const double* __restrict__ x0, const double* __restrict__ aty0,
                               double w, double rho) {
  NIRNAY_GRID_STRIDE(j, n) {
    x[j] = w * ((1.0 + rho) * xt[j] - rho * x[j]) + (1.0 - w) * x0[j];
    aty[j] = w * ((1.0 + rho) * atyt[j] - rho * aty[j]) + (1.0 - w) * aty0[j];
  }
}

__global__ void accumulate(int n, const double* __restrict__ a, double* __restrict__ sum_a,
                           const double* __restrict__ b, double* __restrict__ sum_b) {
  NIRNAY_GRID_STRIDE(i, n) {
    sum_a[i] += a[i];
    sum_b[i] += b[i];
  }
}

__global__ void scale_copy(int n, const double* __restrict__ source, double factor,
                           double* __restrict__ target) {
  NIRNAY_GRID_STRIDE(i, n) target[i] = factor * source[i];
}

__global__ void scale_in_place(int n, double* values, double factor) {
  NIRNAY_GRID_STRIDE(i, n) values[i] *= factor;
}

/// sum (factor a - b)^2, or sum (factor a)^2 when b is null.
__global__ void squared_distance(int n, const double* __restrict__ a, double factor,
                                 const double* __restrict__ b, double* partials) {
  __shared__ double shared[kThreads];
  double total = 0.0;
  NIRNAY_GRID_STRIDE(i, n) {
    const double d = factor * a[i] - (b != nullptr ? b[i] : 0.0);
    total += d * d;
  }
  const double block = block_sum(total, shared);
  if (threadIdx.x == 0) partials[blockIdx.x] = block;
}

// ---- Measurement in original units: see CpuBackend::measure_row / measure_column ----------

__global__ void measure_rows(int m, const double* __restrict__ y, const double* __restrict__ ax,
                             double factor, const double* __restrict__ row_scale,
                             const double* __restrict__ lower_bound,
                             const double* __restrict__ upper_bound, double* partials) {
  __shared__ double shared[kThreads];
  double primal = 0.0;
  double dual = 0.0;
  double support = 0.0;
  double complementarity = 0.0;
  NIRNAY_GRID_STRIDE(i, m) {
    const double lower = lower_bound[i];
    const double upper = upper_bound[i];
    const double activity = factor * ax[i] / row_scale[i];
    double violation = 0.0;
    if (finite_bound(lower)) violation = fmax(violation, lower - activity);
    if (finite_bound(upper)) violation = fmax(violation, activity - upper);
    primal += violation * violation;
    const double yi = factor * y[i] * row_scale[i];
    if (yi > 0.0) {
      if (finite_bound(upper)) {
        support += yi * upper;
      } else {
        dual += yi * yi;
      }
    } else if (yi < 0.0) {
      if (finite_bound(lower)) {
        support += yi * lower;
      } else {
        dual += yi * yi;
      }
    }
    if (lower != upper) {
      const double lower_slack = finite_bound(lower) ? activity - lower : INFINITY;
      const double upper_slack = finite_bound(upper) ? upper - activity : INFINITY;
      complementarity = fmax(complementarity, fabs(yi) * fmin(lower_slack, upper_slack));
    }
  }
  const double p = block_sum(primal, shared);
  const double d = block_sum(dual, shared);
  const double s = block_sum(support, shared);
  const double c = block_max(complementarity, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = p;
    partials[gridDim.x + blockIdx.x] = d;
    partials[2 * gridDim.x + blockIdx.x] = s;
    partials[3 * gridDim.x + blockIdx.x] = c;
  }
}

__global__ void measure_columns(int n, const double* __restrict__ x,
                                const double* __restrict__ aty, double factor,
                                const double* __restrict__ col_scale,
                                const double* __restrict__ cost,
                                const double* __restrict__ lower_bound,
                                const double* __restrict__ upper_bound, double* partials) {
  __shared__ double shared[kThreads];
  double dual = 0.0;
  double bound = 0.0;
  double objective = 0.0;
  double complementarity = 0.0;
  NIRNAY_GRID_STRIDE(j, n) {
    const double lower = lower_bound[j];
    const double upper = upper_bound[j];
    const double xj = factor * x[j] * col_scale[j];
    const double d = cost[j] + factor * aty[j] / col_scale[j];
    if (d > 0.0) {
      if (finite_bound(lower)) {
        bound += d * lower;
      } else {
        dual += d * d;
      }
    } else if (d < 0.0) {
      if (finite_bound(upper)) {
        bound += d * upper;
      } else {
        dual += d * d;
      }
    }
    if (lower != upper) {
      const double lower_slack = finite_bound(lower) ? xj - lower : INFINITY;
      const double upper_slack = finite_bound(upper) ? upper - xj : INFINITY;
      complementarity = fmax(complementarity, fabs(d) * fmin(lower_slack, upper_slack));
    }
    objective += cost[j] * xj;
  }
  const double dd = block_sum(dual, shared);
  const double b = block_sum(bound, shared);
  const double o = block_sum(objective, shared);
  const double c = block_max(complementarity, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = dd;
    partials[gridDim.x + blockIdx.x] = b;
    partials[2 * gridDim.x + blockIdx.x] = o;
    partials[3 * gridDim.x + blockIdx.x] = c;
  }
}

/// Fold `quantities` rows of per-block partials, in block order, into out[0..quantities).
/// Bit q of `max_mask` makes quantity q a maximum instead of a sum.
__global__ void finalize(const double* __restrict__ partials, int blocks, int quantities,
                         unsigned max_mask, double* __restrict__ out) {
  __shared__ double shared[kThreads];
  for (int q = 0; q < quantities; ++q) {
    const bool is_max = ((max_mask >> q) & 1u) != 0u;
    double acc = 0.0;
    for (int b = static_cast<int>(threadIdx.x); b < blocks; b += blockDim.x) {
      const double v = partials[q * blocks + b];
      acc = is_max ? fmax(acc, v) : acc + v;
    }
    const double total = is_max ? block_max(acc, shared) : block_sum(acc, shared);
    if (threadIdx.x == 0) out[q] = total;
  }
}

// ---- Host side ------------------------------------------------------------------------------

/// A device allocation, freed on destruction.
template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept { std::swap(ptr_, other.ptr_); }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    std::swap(ptr_, other.ptr_);
    return *this;
  }
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  cudaError_t allocate(std::size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(&ptr_), std::max<std::size_t>(count, 1) *
                                                           sizeof(T));
  }
  [[nodiscard]] T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

struct Slot {
  double* x = nullptr;    // cols
  double* aty = nullptr;  // cols
  double* y = nullptr;    // rows
  double* ax = nullptr;   // rows
};

struct View {
  const double* x;
  const double* aty;
  const double* y;
  const double* ax;
  double factor;
};

class GpuBackend final : public Backend {
 public:
  GpuBackend(const ScaledProblem& problem, int device)
      : n_(problem.cols), m_(problem.rows), device_(device) {
    if (!ok(cudaSetDevice(device), "cudaSetDevice")) return;
    cudaDeviceProp prop{};
    if (!ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return;
    description_ = std::string("GPU ") + prop.name + ", " +
                   std::to_string(prop.multiProcessorCount) + " SMs, " +
                   std::to_string(prop.totalGlobalMem >> 30) + " GiB";
    max_blocks_ = std::max(1, prop.multiProcessorCount * 8);
    blocks_n_ = grid(n_);
    blocks_m_ = grid(m_);
    if (!ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate")) {
      return;
    }
    if (!sparse_ok(cusparseCreate(&handle_), "cusparseCreate")) return;
    if (!sparse_ok(cusparseSetStream(handle_, stream_), "cusparseSetStream")) return;
    if (!upload(problem)) return;
    if (!allocate_vectors()) return;
    if (!build_sparse(problem)) return;
    ready_ = true;
  }

  ~GpuBackend() override {
    if (mat_a_ != nullptr) cusparseDestroySpMat(mat_a_);
    if (mat_at_ != nullptr) cusparseDestroySpMat(mat_at_);
    for (cusparseDnVecDescr_t v : {vec_n_in_, vec_m_out_, vec_m_in_, vec_n_out_}) {
      if (v != nullptr) cusparseDestroyDnVec(v);
    }
    if (handle_ != nullptr) cusparseDestroy(handle_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  GpuBackend(const GpuBackend&) = delete;
  GpuBackend& operator=(const GpuBackend&) = delete;

  [[nodiscard]] bool ready() const { return ready_; }

  [[nodiscard]] std::string name() const override { return "pdhg-gpu"; }
  [[nodiscard]] std::string description() const override { return description_; }
  [[nodiscard]] std::string failure() const override { return failure_; }
  [[nodiscard]] Count averaged() const override { return averaged_; }

  [[nodiscard]] double spectral_norm(int iterations, unsigned seed) override {
    if (!healthy() || m_ == 0 || n_ == 0 || nonzeros_ == 0) return 1.0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> spread(-1.0, 1.0);
    std::vector<double> host(static_cast<std::size_t>(n_));
    for (double& value : host) value = spread(rng);
    double* v = output_.x;  // scratch until initialize() overwrites it
    double* av = output_.ax;
    copy_to_device(v, host.data(), host.size());
    double norm = 0.0;
    for (int step = 0; step < iterations && healthy(); ++step) {
      const double length = std::sqrt(sum_of_squares(v, n_, 1.0));
      if (length == 0.0) return 1.0;
      scale_in_place<<<blocks_n_, kThreads, 0, stream_>>>(n_, v, 1.0 / length);
      multiply_a(v, av);
      multiply_at(av, v);
      norm = std::sqrt(std::sqrt(sum_of_squares(v, n_, 1.0)));
    }
    return norm > 0.0 ? norm * 1.01 : 1.0;
  }

  void initialize(const std::vector<double>& x, const std::vector<double>& y) override {
    if (!healthy()) return;
    copy_to_device(iterate_.x, x.data(), x.size());
    copy_to_device(iterate_.y, y.data(), y.size());
    multiply_a(iterate_.x, iterate_.ax);
    multiply_at(iterate_.y, iterate_.aty);
    copy_slot(iterate_, &anchor_);
    copy_slot(iterate_, &best_);
    copy_slot(iterate_, &output_);
    clear_average();
  }

  StepMeasure trial_step(double tau, double sigma) override {
    if (!healthy()) return {};
    primal_output<true><<<blocks_n_, kThreads, 0, stream_>>>(
        n_, iterate_.x, iterate_.aty, cost_.get(), lower_.get(), upper_.get(), tau, output_.x,
        partials_primal_.get());
    fold(partials_primal_.get(), blocks_n_, 1, 0u, results_.get());
    multiply_a(output_.x, output_.ax);
    dual_trial<<<blocks_m_, kThreads, 0, stream_>>>(m_, iterate_.y, iterate_.ax, output_.ax,
                                                    row_lower_.get(), row_upper_.get(), sigma,
                                                    output_.y, partials_dual_.get());
    fold(partials_dual_.get(), blocks_m_, 2, 0u, results_.get() + 1);
    double host[3] = {0.0, 0.0, 0.0};
    read_results(host, 3);
    return {host[0], host[1], host[2]};
  }

  void accept_trial() override {
    if (!healthy()) return;
    std::swap(iterate_.x, output_.x);
    std::swap(iterate_.y, output_.y);
    std::swap(iterate_.ax, output_.ax);
    multiply_at(iterate_.y, iterate_.aty);
    accumulate<<<blocks_n_, kThreads, 0, stream_>>>(n_, iterate_.x, sum_.x, iterate_.aty,
                                                    sum_.aty);
    accumulate<<<blocks_m_, kThreads, 0, stream_>>>(m_, iterate_.y, sum_.y, iterate_.ax,
                                                    sum_.ax);
    ++averaged_;
  }

  void halpern_steps(Count count, Count first_k, double tau, double sigma,
                     double reflection) override {
    if (!healthy()) return;
    for (Count s = 0; s < count; ++s) {
      const auto k = static_cast<double>(first_k + s);
      const double w = (k + 1.0) / (k + 2.0);
      const bool last = s + 1 == count;
      if (last) {
        primal_output<true><<<blocks_n_, kThreads, 0, stream_>>>(
            n_, iterate_.x, iterate_.aty, cost_.get(), lower_.get(), upper_.get(), tau,
            output_.x, partials_primal_.get());
      } else {
        primal_output<false><<<blocks_n_, kThreads, 0, stream_>>>(
            n_, iterate_.x, iterate_.aty, cost_.get(), lower_.get(), upper_.get(), tau,
            output_.x, nullptr);
      }
      multiply_a(output_.x, output_.ax);
      if (last) {
        dual_halpern<true><<<blocks_m_, kThreads, 0, stream_>>>(
            m_, iterate_.y, iterate_.ax, output_.ax, row_lower_.get(), row_upper_.get(), sigma,
            w, reflection, anchor_.y, anchor_.ax, output_.y, partials_dual_.get());
      } else {
        dual_halpern<false><<<blocks_m_, kThreads, 0, stream_>>>(
            m_, iterate_.y, iterate_.ax, output_.ax, row_lower_.get(), row_upper_.get(), sigma,
            w, reflection, anchor_.y, anchor_.ax, output_.y, nullptr);
      }
      multiply_at(output_.y, output_.aty);
      primal_halpern<<<blocks_n_, kThreads, 0, stream_>>>(n_, iterate_.x, iterate_.aty,
                                                          output_.x, output_.aty, anchor_.x,
                                                          anchor_.aty, w, reflection);
    }
    if (count > 0) {
      fold(partials_primal_.get(), blocks_n_, 1, 0u, results_.get());
      fold(partials_dual_.get(), blocks_m_, 2, 0u, results_.get() + 1);
      double host[3] = {0.0, 0.0, 0.0};
      read_results(host, 3);
      fixed_point_ = {host[0], host[1], host[2]};
    }
  }

  [[nodiscard]] StepMeasure fixed_point_measure() const override { return fixed_point_; }

  [[nodiscard]] KktSums measure(Point point) override {
    if (!healthy()) return {};
    const View v = view(point);
    measure_rows<<<blocks_m_, kThreads, 0, stream_>>>(m_, v.y, v.ax, v.factor, row_scale_.get(),
                                                      orig_row_lower_.get(),
                                                      orig_row_upper_.get(),
                                                      partials_dual_.get());
    fold(partials_dual_.get(), blocks_m_, 4, 0b1000u, results_.get());
    measure_columns<<<blocks_n_, kThreads, 0, stream_>>>(
        n_, v.x, v.aty, v.factor, col_scale_.get(), orig_cost_.get(), orig_lower_.get(),
        orig_upper_.get(), partials_primal_.get());
    fold(partials_primal_.get(), blocks_n_, 4, 0b1000u, results_.get() + 4);
    double host[8] = {};
    read_results(host, 8);
    KktSums sums;
    sums.primal_violation_squared = host[0];
    sums.dual_violation_squared = host[4] + host[1];
    sums.support = host[2];
    sums.bound_contribution = host[5];
    sums.primal_objective = host[6];
    sums.complementarity = std::max(host[3], host[7]);
    return sums;
  }

  [[nodiscard]] StepMeasure distance_to_anchor(Point point) override {
    if (!healthy()) return {};
    const View v = view(point);
    squared_distance<<<blocks_n_, kThreads, 0, stream_>>>(n_, v.x, v.factor, anchor_.x,
                                                          partials_primal_.get());
    fold(partials_primal_.get(), blocks_n_, 1, 0u, results_.get());
    squared_distance<<<blocks_m_, kThreads, 0, stream_>>>(m_, v.y, v.factor, anchor_.y,
                                                          partials_dual_.get());
    fold(partials_dual_.get(), blocks_m_, 1, 0u, results_.get() + 1);
    double host[2] = {0.0, 0.0};
    read_results(host, 2);
    return {host[0], host[1], 0.0};
  }

  void restart_to(Point point) override {
    if (!healthy()) return;
    if (point != Point::kIterate) copy_view(view(point), &iterate_);
    copy_slot(iterate_, &anchor_);
    clear_average();
  }

  void save_best(Point point) override {
    if (!healthy()) return;
    copy_view(view(point), &best_);
  }

  void download(Point point, std::vector<double>* x, std::vector<double>* y) override {
    x->assign(static_cast<std::size_t>(n_), 0.0);
    y->assign(static_cast<std::size_t>(m_), 0.0);
    if (!healthy()) return;
    const View v = view(point);
    const double* xs = v.x;
    const double* ys = v.y;
    if (v.factor != 1.0) {
      scale_copy<<<blocks_n_, kThreads, 0, stream_>>>(n_, v.x, v.factor, scratch_.x);
      scale_copy<<<blocks_m_, kThreads, 0, stream_>>>(m_, v.y, v.factor, scratch_.y);
      xs = scratch_.x;
      ys = scratch_.y;
    }
    ok(cudaMemcpyAsync(x->data(), xs, x->size() * sizeof(double), cudaMemcpyDeviceToHost,
                       stream_),
       "download x");
    ok(cudaMemcpyAsync(y->data(), ys, y->size() * sizeof(double), cudaMemcpyDeviceToHost,
                       stream_),
       "download y");
    ok(cudaStreamSynchronize(stream_), "synchronize after download");
  }

 private:
  [[nodiscard]] bool healthy() const { return ready_ && failure_.empty(); }

  bool ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    if (failure_.empty()) {
      failure_ = std::string(what) + ": " + cudaGetErrorString(status);
    }
    return false;
  }

  bool sparse_ok(cusparseStatus_t status, const char* what) {
    if (status == CUSPARSE_STATUS_SUCCESS) return true;
    if (failure_.empty()) {
      failure_ = std::string(what) + ": " + cusparseGetErrorString(status);
    }
    return false;
  }

  [[nodiscard]] int grid(int size) const {
    const int needed = (size + kThreads - 1) / kThreads;
    return std::clamp(needed, 1, max_blocks_);
  }

  template <typename T>
  bool upload_array(DeviceArray<T>* target, const T* source, std::size_t count,
                    const char* what) {
    if (!ok(target->allocate(count), what)) return false;
    if (count == 0) return true;
    return ok(cudaMemcpy(target->get(), source, count * sizeof(T), cudaMemcpyHostToDevice), what);
  }

  bool upload(const ScaledProblem& p) {
    const auto n = static_cast<std::size_t>(n_);
    const auto m = static_cast<std::size_t>(m_);
    nonzeros_ = p.by_rows.starts[m_];
    const auto nnz = static_cast<std::size_t>(nonzeros_);
    return upload_array(&a_starts_, p.by_rows.starts, m + 1, "upload A") &&
           upload_array(&a_indices_, p.by_rows.indices, nnz, "upload A") &&
           upload_array(&a_values_, p.by_rows.values, nnz, "upload A") &&
           upload_array(&at_starts_, p.by_columns.starts, n + 1, "upload A'") &&
           upload_array(&at_indices_, p.by_columns.indices, nnz, "upload A'") &&
           upload_array(&at_values_, p.by_columns.values, nnz, "upload A'") &&
           upload_array(&cost_, p.cost, n, "upload c") &&
           upload_array(&lower_, p.col_lower, n, "upload l") &&
           upload_array(&upper_, p.col_upper, n, "upload u") &&
           upload_array(&row_lower_, p.row_lower, m, "upload row bounds") &&
           upload_array(&row_upper_, p.row_upper, m, "upload row bounds") &&
           upload_array(&col_scale_, p.col_scale, n, "upload scaling") &&
           upload_array(&row_scale_, p.row_scale, m, "upload scaling") &&
           upload_array(&orig_cost_, p.original_cost, n, "upload original c") &&
           upload_array(&orig_lower_, p.original_col_lower, n, "upload original bounds") &&
           upload_array(&orig_upper_, p.original_col_upper, n, "upload original bounds") &&
           upload_array(&orig_row_lower_, p.original_row_lower, m, "upload original bounds") &&
           upload_array(&orig_row_upper_, p.original_row_upper, m, "upload original bounds");
  }

  bool allocate_vectors() {
    const auto n = static_cast<std::size_t>(n_);
    const auto m = static_cast<std::size_t>(m_);
    // Six slots of four vectors: iterate, output, anchor, best, average sums, scratch.
    storage_.resize(6 * 4);
    Slot* slots[] = {&iterate_, &output_, &anchor_, &best_, &sum_, &scratch_};
    std::size_t next = 0;
    for (Slot* slot : slots) {
      double** fields[] = {&slot->x, &slot->aty, &slot->y, &slot->ax};
      const std::size_t sizes[] = {n, n, m, m};
      for (int f = 0; f < 4; ++f) {
        if (!ok(storage_[next].allocate(sizes[f]), "allocate vectors")) return false;
        *fields[f] = storage_[next].get();
        ++next;
      }
    }
    const auto slots_needed = static_cast<std::size_t>(kMaxQuantities * max_blocks_);
    return ok(partials_primal_.allocate(slots_needed), "allocate partials") &&
           ok(partials_dual_.allocate(slots_needed), "allocate partials") &&
           ok(results_.allocate(8), "allocate results");
  }

  bool build_sparse(const ScaledProblem& p) {
    (void)p;
    if (!sparse_ok(cusparseCreateCsr(&mat_a_, m_, n_, nonzeros_, a_starts_.get(),
                                     a_indices_.get(), a_values_.get(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                   "cusparseCreateCsr(A)") ||
        !sparse_ok(cusparseCreateCsr(&mat_at_, n_, m_, nonzeros_, at_starts_.get(),
                                     at_indices_.get(), at_values_.get(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                   "cusparseCreateCsr(A')") ||
        !sparse_ok(cusparseCreateDnVec(&vec_n_in_, n_, iterate_.x, CUDA_R_64F), "vector") ||
        !sparse_ok(cusparseCreateDnVec(&vec_m_out_, m_, iterate_.ax, CUDA_R_64F), "vector") ||
        !sparse_ok(cusparseCreateDnVec(&vec_m_in_, m_, iterate_.y, CUDA_R_64F), "vector") ||
        !sparse_ok(cusparseCreateDnVec(&vec_n_out_, n_, iterate_.aty, CUDA_R_64F), "vector")) {
      return false;
    }
    const double one = 1.0;
    const double zero = 0.0;
    std::size_t size_a = 0;
    std::size_t size_at = 0;
    if (!sparse_ok(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                           mat_a_, vec_n_in_, &zero, vec_m_out_, CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2, &size_a),
                   "cusparseSpMV_bufferSize(A)") ||
        !sparse_ok(cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                           mat_at_, vec_m_in_, &zero, vec_n_out_, CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2, &size_at),
                   "cusparseSpMV_bufferSize(A')")) {
      return false;
    }
    return ok(buffer_a_.allocate(size_a), "allocate SpMV buffer") &&
           ok(buffer_at_.allocate(size_at), "allocate SpMV buffer");
  }

  void spmv(cusparseSpMatDescr_t mat, cusparseDnVecDescr_t in, cusparseDnVecDescr_t out,
            const double* x, double* y, void* buffer) {
    if (!healthy()) return;
    const double one = 1.0;
    const double zero = 0.0;
    if (!sparse_ok(cusparseDnVecSetValues(in, const_cast<double*>(x)), "set vector") ||
        !sparse_ok(cusparseDnVecSetValues(out, y), "set vector")) {
      return;
    }
    sparse_ok(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, in, &zero, out,
                           CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, buffer),
              "cusparseSpMV");
  }

  void multiply_a(const double* x, double* ax) {
    if (m_ == 0) return;
    spmv(mat_a_, vec_n_in_, vec_m_out_, x, ax, buffer_a_.get());
  }

  void multiply_at(const double* y, double* aty) {
    if (n_ == 0) return;
    if (m_ == 0) {
      ok(cudaMemsetAsync(aty, 0, static_cast<std::size_t>(n_) * sizeof(double), stream_),
         "memset");
      return;
    }
    spmv(mat_at_, vec_m_in_, vec_n_out_, y, aty, buffer_at_.get());
  }

  void fold(const double* partials, int blocks, int quantities, unsigned max_mask, double* out) {
    finalize<<<1, kThreads, 0, stream_>>>(partials, blocks, quantities, max_mask, out);
  }

  void read_results(double* host, int count) {
    ok(cudaGetLastError(), "kernel launch");
    ok(cudaMemcpyAsync(host, results_.get(), static_cast<std::size_t>(count) * sizeof(double),
                       cudaMemcpyDeviceToHost, stream_),
       "read results");
    ok(cudaStreamSynchronize(stream_), "synchronize");
  }

  double sum_of_squares(const double* v, int size, double factor) {
    squared_distance<<<grid(size), kThreads, 0, stream_>>>(size, v, factor, nullptr,
                                                           partials_primal_.get());
    fold(partials_primal_.get(), grid(size), 1, 0u, results_.get());
    double host = 0.0;
    read_results(&host, 1);
    return host;
  }

  void copy_to_device(double* target, const double* source, std::size_t count) {
    if (count == 0) return;
    ok(cudaMemcpyAsync(target, source, count * sizeof(double), cudaMemcpyHostToDevice, stream_),
       "upload vector");
    ok(cudaStreamSynchronize(stream_), "synchronize after upload");
  }

  void copy_device(double* target, const double* source, int count) {
    if (count == 0 || target == source) return;
    ok(cudaMemcpyAsync(target, source, static_cast<std::size_t>(count) * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream_),
       "copy vector");
  }

  void copy_slot(const Slot& from, Slot* to) {
    copy_device(to->x, from.x, n_);
    copy_device(to->aty, from.aty, n_);
    copy_device(to->y, from.y, m_);
    copy_device(to->ax, from.ax, m_);
  }

  void copy_view(const View& v, Slot* to) {
    if (v.factor == 1.0) {
      copy_device(to->x, v.x, n_);
      copy_device(to->aty, v.aty, n_);
      copy_device(to->y, v.y, m_);
      copy_device(to->ax, v.ax, m_);
      return;
    }
    scale_copy<<<blocks_n_, kThreads, 0, stream_>>>(n_, v.x, v.factor, to->x);
    scale_copy<<<blocks_n_, kThreads, 0, stream_>>>(n_, v.aty, v.factor, to->aty);
    scale_copy<<<blocks_m_, kThreads, 0, stream_>>>(m_, v.y, v.factor, to->y);
    scale_copy<<<blocks_m_, kThreads, 0, stream_>>>(m_, v.ax, v.factor, to->ax);
  }

  void clear_average() {
    const auto n_bytes = static_cast<std::size_t>(n_) * sizeof(double);
    const auto m_bytes = static_cast<std::size_t>(m_) * sizeof(double);
    ok(cudaMemsetAsync(sum_.x, 0, n_bytes, stream_), "clear average");
    ok(cudaMemsetAsync(sum_.aty, 0, n_bytes, stream_), "clear average");
    ok(cudaMemsetAsync(sum_.y, 0, m_bytes, stream_), "clear average");
    ok(cudaMemsetAsync(sum_.ax, 0, m_bytes, stream_), "clear average");
    averaged_ = 0;
  }

  [[nodiscard]] View view(Point point) const {
    switch (point) {
      case Point::kIterate: return {iterate_.x, iterate_.aty, iterate_.y, iterate_.ax, 1.0};
      case Point::kOutput: return {output_.x, output_.aty, output_.y, output_.ax, 1.0};
      case Point::kAverage:
        if (averaged_ > 0) {
          return {sum_.x, sum_.aty, sum_.y, sum_.ax, 1.0 / static_cast<double>(averaged_)};
        }
        return view(Point::kIterate);
      case Point::kBest: return {best_.x, best_.aty, best_.y, best_.ax, 1.0};
    }
    return view(Point::kIterate);
  }

  const int n_;
  const int m_;
  const int device_;
  int nonzeros_ = 0;
  int max_blocks_ = 1;
  int blocks_n_ = 1;
  int blocks_m_ = 1;
  bool ready_ = false;
  std::string failure_;
  std::string description_;

  cudaStream_t stream_ = nullptr;
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t mat_a_ = nullptr;
  cusparseSpMatDescr_t mat_at_ = nullptr;
  cusparseDnVecDescr_t vec_n_in_ = nullptr;
  cusparseDnVecDescr_t vec_m_out_ = nullptr;
  cusparseDnVecDescr_t vec_m_in_ = nullptr;
  cusparseDnVecDescr_t vec_n_out_ = nullptr;
  DeviceArray<char> buffer_a_;
  DeviceArray<char> buffer_at_;

  DeviceArray<Index> a_starts_, a_indices_, at_starts_, at_indices_;
  DeviceArray<double> a_values_, at_values_;
  DeviceArray<double> cost_, lower_, upper_, row_lower_, row_upper_;
  DeviceArray<double> col_scale_, row_scale_;
  DeviceArray<double> orig_cost_, orig_lower_, orig_upper_, orig_row_lower_, orig_row_upper_;
  std::vector<DeviceArray<double>> storage_;
  DeviceArray<double> partials_primal_, partials_dual_, results_;

  Slot iterate_, output_, anchor_, best_, sum_, scratch_;
  Count averaged_ = 0;
  StepMeasure fixed_point_;
};

}  // namespace

std::unique_ptr<Backend> make_gpu_backend(const ScaledProblem& problem, int device,
                                          std::string* why) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    if (why != nullptr) {
      *why = status != cudaSuccess
                 ? std::string("no CUDA device answers (") + cudaGetErrorString(status) + ")"
                 : std::string("no CUDA device is visible");
    }
    return nullptr;
  }
  if (device < 0 || device >= count) {
    if (why != nullptr) {
      *why = "gpu_device " + std::to_string(device) + " does not exist (" +
             std::to_string(count) + " visible)";
    }
    return nullptr;
  }
  auto backend = std::make_unique<GpuBackend>(problem, device);
  if (!backend->ready()) {
    if (why != nullptr) *why = "the CUDA backend could not start: " + backend->failure();
    return nullptr;
  }
  return backend;
}

}  // namespace nirnay::pdhg

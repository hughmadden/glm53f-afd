// L2 prefetch of weight ranges: while a MoE layer's routed experts are out on the ranks, the
// decode loop pulls the start of the next layer's weights into L2 on a side stream
// (src/prefetch.rs).
//
// Two ways to touch a range every `stride` bytes (GLM53F_FWD_PREFETCH_*):
// - LOAD: a one-byte load through L2 only (`ld.global.cg`, with the `.L2::256B` prefetch size),
//   four in flight per thread, the values folded into a register that is never stored (a store
//   to a null sink the compiler cannot rule out keeps the loads);
// - HINT: `prefetch.global.L2`, which the memory system may drop (on the RTX 4090 most are:
//   `examples/l2_prefetch_bench.rs`).
// Every address touched lies inside its range. Nothing is written, so a prefetch changes no
// result: the later loads read the same bytes, from L2 instead of DRAM while the lines are still
// there.
#include "glm53f_forward.h"
#include "common.cuh"

namespace glm53f_fwd {
namespace {

constexpr int kMaxRanges = GLM53F_FWD_PREFETCH_MAX_RANGES;
constexpr int kUnroll = 4;

// The ranges, passed by value (the kernel's parameter space).
struct Ranges {
  const char* ptr[kMaxRanges];
  int64_t lines[kMaxRanges];
  int n;
  int stride;
  // Always null (see above).
  uint32_t* sink;
};

__device__ __forceinline__ uint32_t load_cg(const char* p) {
  uint32_t v;
  asm volatile("ld.global.cg.L2::256B.u8 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

template <int Mode>
__global__ void __launch_bounds__(256) l2_prefetch_kernel(const Ranges r) {
  const int64_t step = int64_t(gridDim.x) * blockDim.x;
  const int64_t t0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t acc = 0;
  for (int s = 0; s < r.n; ++s) {
    const char* base = r.ptr[s];
    const int64_t lines = r.lines[s];
    if (Mode == GLM53F_FWD_PREFETCH_HINT) {
      for (int64_t i = t0; i < lines; i += step) asm volatile("prefetch.global.L2 [%0];" ::"l"(base + i * r.stride));
      continue;
    }
    int64_t i = t0;
    for (; i + (kUnroll - 1) * step < lines; i += kUnroll * step) {
      uint32_t v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = load_cg(base + (i + u * step) * r.stride);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) acc ^= v[u];
    }
    for (; i < lines; i += step) acc ^= load_cg(base + i * r.stride);
  }
  if (r.sink) *r.sink = acc;
}

}  // namespace
}  // namespace glm53f_fwd

using namespace glm53f_fwd;

extern "C" int32_t glm53f_fwd_l2_prefetch(const void* const* ptrs, const int64_t* bytes, int32_t n, int32_t stride,
                                          int32_t blocks, int32_t mode, cudaStream_t stream) {
  if (n < 0 || n > kMaxRanges || blocks < 1 || (stride != 32 && stride != 64 && stride != 128 && stride != 256) ||
      (mode != GLM53F_FWD_PREFETCH_LOAD && mode != GLM53F_FWD_PREFETCH_HINT) || (n > 0 && (!ptrs || !bytes)))
    return cudaErrorInvalidValue;
  Ranges r{};
  int k = 0;
  for (int i = 0; i < n; ++i) {
    if (!ptrs[i] || bytes[i] < 0) return cudaErrorInvalidValue;
    if (bytes[i] == 0) continue;
    r.ptr[k] = static_cast<const char*>(ptrs[i]);
    // Touches at 0, stride, 2 stride, ...: every one inside the range.
    r.lines[k] = (bytes[i] + stride - 1) / stride;
    ++k;
  }
  if (k == 0) return cudaSuccess;
  r.n = k;
  r.stride = stride;
  if (mode == GLM53F_FWD_PREFETCH_LOAD)
    l2_prefetch_kernel<GLM53F_FWD_PREFETCH_LOAD><<<blocks, 256, 0, stream>>>(r);
  else
    l2_prefetch_kernel<GLM53F_FWD_PREFETCH_HINT><<<blocks, 256, 0, stream>>>(r);
  return cudaGetLastError();
}

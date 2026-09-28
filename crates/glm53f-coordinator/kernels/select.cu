/* Row selection kernels for the coordinator's sampler: the device argmax and the per-row token
 * masks.
 *
 * `argmax_kernel` and its two entry points are copied from mimo26f-afd v1.2.0
 * `crates/mimo26-coordinator/kernels/dflash.cu` (the argmax part; the drafter's kernels stay
 * there); only the entry points' names changed (PROVENANCE.md). Called with `vocab` = the
 * tokenizer's id bound, so the LM head's padding rows are never the argmax.
 *
 * `mask_rows_kernel` is new: a grammar's allowed-token bitset per row (null for none) sets every
 * other logit to -inf before the argmax and the draw (`src/sampling.rs` `apply_mask` is its CPU
 * reference).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

/* First index of the row maximum (the host `greedy` rule; NaN never wins).
 * With `prob`, also the softmax probability of that maximum (1 / sum exp(x - max)). */
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int64_t ld, int vocab,
                                                      int* __restrict__ out, float* __restrict__ prob) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const float* row = x + (int64_t)blockIdx.x * ld;
  float best = -INFINITY;
  int besti = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = row[i];
    if (v > best || (v == best && i < besti)) {
      best = v;
      besti = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
    if (ov > best || (ov == best && oi < besti)) {
      best = ov;
      besti = oi;
    }
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    bv[warp] = best;
    bi[warp] = besti;
  }
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    best = lane < nw ? bv[lane] : -INFINITY;
    besti = lane < nw ? bi[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
      if (ov > best || (ov == best && oi < besti)) {
        best = ov;
        besti = oi;
      }
    }
    if (lane == 0) {
      out[blockIdx.x] = besti == 0x7fffffff ? 0 : besti;
      bv[0] = best;
    }
  }
  if (!prob) return;
  __syncthreads();
  const float mx = bv[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = row[i];
    if (v == v) sum += __expf(v - mx);
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) bv[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    sum = lane < nw ? bv[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) prob[blockIdx.x] = sum > 0.f ? 1.f / sum : 0.f;
  }
}


/* Row r (stride ld, `cols` columns): every logit whose token masks[r] does not allow becomes -inf.
 * masks[r] is a bitset (bit i % 32 of word i / 32 set: token i allowed) of `bits` tokens, or null
 * (the row is left alone). Tokens at or past `bits` are not allowed. One block per row. */
__global__ void __launch_bounds__(256) mask_rows_kernel(float* __restrict__ x, int64_t ld, int cols,
                                                        const uint32_t* const* __restrict__ masks, int bits) {
  const uint32_t* m = masks[blockIdx.x];
  if (m == nullptr) return;
  float* row = x + (int64_t)blockIdx.x * ld;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const bool allowed = i < bits && ((m[i >> 5] >> (i & 31)) & 1u);
    if (!allowed) row[i] = -INFINITY;
  }
}

} /* namespace */

extern "C" cudaError_t glm53f_coord_argmax_rows(const float* x, int64_t ld, int rows, int vocab, int* out, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  argmax_kernel<<<rows, 1024, 0, s>>>(x, ld, vocab, out, nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t glm53f_coord_argmax_prob_rows(const float* x, int64_t ld, int rows, int vocab, int* out, float* prob,
                                                    cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  argmax_kernel<<<rows, 1024, 0, s>>>(x, ld, vocab, out, prob);
  return cudaGetLastError();
}

extern "C" cudaError_t glm53f_coord_mask_rows(float* x, int64_t ld, int rows, int cols, const uint32_t* const* masks,
                                              int bits, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  if (cols < 0 || bits < 0 || (int64_t)cols > ld) return cudaErrorInvalidValue;
  mask_rows_kernel<<<rows, 256, 0, s>>>(x, ld, cols, masks, bits);
  return cudaGetLastError();
}

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright the v100-skinny contributors
//
// Fused NVFP4 GEMM + greedy argmax for the SM70 lm_head top-1 path.
//
// Ported from dnv2003/v100-skinny (MIT) — see LICENSE.v100-skinny in this
// directory. This is skinny's `skinny_nvfp4_simt<1, KC, R, ARGMAX=true>`
// plus its TurboMind-derived e2m1 decoder and the fp8-e4m3 group-scale
// decoder, extracted as a standalone sidecar so the fused argmax can ship
// without relinking the primary vllm._C module (same pattern as
// sm70_top20_philox_sampler).
//
// Computes y[n] = x[0] . W[n] for every vocab row n (M=1) and reduces to
// the per-block max in registers; one (value, index) pair per block, no
// logits tensor ever reaches HBM. Ties keep the LOWEST index, matching
// argmax-over-half semantics of the separate top-1 path.
//
// Layout contract (modelopt NVFP4 checkpoints, e.g. Swift-1.5-27B):
//   codes  uint8 [V, K/2]  two e2m1 codes per byte, low nibble = even k
//   scales uint8-as-fp8 [V, K/16]  fp8-e4m3 per 16-k group
//   gscale float  global scale (checkpoint's weight_scale_2)
//   value  = e2m1_code * fp8_scale * gscale

#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define DEV_INLINE __device__ __forceinline__

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the
// simt read pattern (lane-groups sharing a bank base differ in p>>5).
DEV_INLINE int swz(int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); }

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// e2m1 decoder derived from TurboMind's cvt_f16x8_e2m1 (Apache-2.0;
// 1Cat-vLLM csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/attention/
// quantization.h). Shifts sign/EM bits into fp16 positions; the 2^14
// exponent re-bias is folded into the caller's scale, so no extra
// multiply. Output half2 pairing is INTERLEAVED: out[i] holds codes
// (i, i+4) of the 8-code word.
DEV_INLINE void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2p);
}

// Stage 8 contiguous activation halves as four half2 pairs with the
// (k, k+4) pairing dequant8_tm's interleaved output needs.
DEV_INLINE void stage_pairs(half2 *dst, int base_pair, const uint4 &v) {
  const unsigned *r = reinterpret_cast<const unsigned *>(&v);
  unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                   __byte_perm(r[0], r[2], 0x7632),
                   __byte_perm(r[1], r[3], 0x5410),
                   __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; j++)
    dst[swz(base_pair + j)] = *reinterpret_cast<half2 *>(&o[j]);
}

// SIMT kernel: 8 warps/block, one output row per warp, M=1 (decode).
// ARGMAX writes one (half value, int index) pair per block.
template <int KC, int R>
__global__ void skinny_nvfp4_simt_argmax(
    const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
    const half *__restrict__ x, int N, int K, float gscale,
    half *__restrict__ bvals, int *__restrict__ bidxs) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [1][KC/2] swizzled
  constexpr int P2 = KC / 2;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (blockIdx.x * 8 + warp) * R;
  const uint8_t *crow[R];
  const uint8_t *srow[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    crow[r] = codes + (size_t)(n + r) * (K >> 1);
    srow[r] = scales + (size_t)(n + r) * (K >> 4);
  }
  // Fold the global scale into the group scales so in-kernel weights sit
  // at their true O(0.1) magnitudes; otherwise code*fp8scale reaches
  // ~2.7e3 and fp16 products overflow on real activation outliers.
  // dequant8_tm needs a 2^14 exponent re-bias; fold it here for free.
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  float accf[R] = {0.f};

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < (KC / 8); idx += blockDim.x) {
      const int j4 = idx;
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + k0 + j4 * 8);
      stage_pairs(xs, j4 * 4, v);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < KC / 512; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      // fp16 accumulation window is one 16-code segment (8 products per
      // half2 lane); flushed to fp32 so real activation outliers cannot
      // overflow half range.
      half2 acch[R];
#pragma unroll
      for (int r = 0; r < R; r++) acch[r] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
          dequant8_tm(qw, sc2[r], w4[r]);
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
          const half2 xv = xs[psw];
#pragma unroll
          for (int r = 0; r < R; r++)
            acch[r] = __hfma2(w4[r][pi], xv, acch[r]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float2 f = __half22float2(acch[r]);
        accf[r] += f.x + f.y;
      }
    }
  }

  // Tail chunk: K % KC remainder (any multiple of 128). Same layout and
  // swizzle, runtime segment bound with idle-lane guard.
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < (tail / 8); idx += blockDim.x) {
      const int j4 = idx;
      const uint4 v = *reinterpret_cast<const uint4 *>(x + k0 + j4 * 8);
      stage_pairs(xs, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      half2 acch[R];
#pragma unroll
      for (int r = 0; r < R; r++) acch[r] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
          dequant8_tm(qw, sc2[r], w4[r]);
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
          const half2 xv = xs[psw];
#pragma unroll
          for (int r = 0; r < R; r++)
            acch[r] = __hfma2(w4[r][pi], xv, acch[r]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float2 f = __half22float2(acch[r]);
        accf[r] += f.x + f.y;
      }
    }
  }

  // Fused greedy argmax (M=1): identical reduce, identical __float2half
  // rounding, then compare halfs with strict > so ties keep the LOWEST
  // index — matching argmax-over-half semantics of the separate path.
  __shared__ half wval[8];
  __shared__ int widx[8];
  half best_h = __float2half(-3.0e38f);
  int best_i = -1;
#pragma unroll
  for (int r = 0; r < R; r++) {
    float v = accf[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (lane == 0) {
      const half h = __float2half(v);
      if (__hgt(h, best_h)) {
        best_h = h;
        best_i = n + r;
      }
    }
  }
  if (lane == 0) {
    wval[warp] = best_h;
    widx[warp] = best_i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    half bh = wval[0];
    int bi = widx[0];
#pragma unroll
    for (int w = 1; w < 8; w++)
      if (__hgt(wval[w], bh)) {
        bh = wval[w];
        bi = widx[w];
      }
    bvals[blockIdx.x] = bh;
    bidxs[blockIdx.x] = bi;
  }
}

std::vector<torch::Tensor> sm70_nvfp4_simt_argmax_out(
    torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
    double gscale) {
  // M=1 fused argmax over the packed vocab rows.
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "M=1 only");
  TORCH_CHECK(x.dtype() == torch::kHalf, "fp16 activations only");
  TORCH_CHECK(codes.dtype() == torch::kUInt8, "codes must be uint8");
  TORCH_CHECK(scales.is_contiguous() && codes.is_contiguous(),
              "codes/scales must be contiguous");
  const int64_t V = codes.size(0);
  const int64_t K2 = codes.size(1);
  const int64_t K = K2 * 2;
  TORCH_CHECK(scales.size(0) == V && scales.size(1) == K / 16,
              "scale shape mismatch");
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
  TORCH_CHECK(V % 8 == 0, "V must be a multiple of 8");

  constexpr int KC = 1024;
  const bool two_rows = (K <= 2048) && (V % 16 == 0);
  const int nblocks = two_rows ? (int)V / 16 : (int)V / 8;
  auto bvals = torch::empty({nblocks}, x.options());
  auto bidxs = torch::empty({nblocks}, x.options().dtype(torch::kInt32));
  const dim3 grid(nblocks), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (KC / 2) * sizeof(half2);

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  if (two_rows)
    skinny_nvfp4_simt_argmax<KC, 2><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), (int)V,
        (int)K, (float)gscale, reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  else
    skinny_nvfp4_simt_argmax<KC, 1><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), (int)V,
        (int)K, (float)gscale, reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  return {bvals, bidxs};
}

// Sidecar registration: adding the fused argmax must not relink or
// perturb the primary vllm._C module; Python loads this fragment only
// for the explicitly admitted native-lm_head route.
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sm70_nvfp4_simt_argmax_out(Tensor x, Tensor codes, Tensor scales, "
      "float gscale) -> Tensor[]");
  ops.impl("sm70_nvfp4_simt_argmax_out", torch::kCUDA,
           &sm70_nvfp4_simt_argmax_out);
}
// Standalone-load export: torch's cpp_extension load() requires the
// module init symbol when building this file outside the wheel build.
// Guarded: the CMake sidecar target provides its own module entry via
// USE_SABI and does not define TORCH_EXTENSION_NAME.
#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &sm70_nvfp4_simt_argmax_out);
}
#endif

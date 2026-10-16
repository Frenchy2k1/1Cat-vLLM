# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from v100-skinny fork_patches/marlin.py (1Cat-vLLM 1.2.2 base),
# 2026-10-16. Adds an optional custom SM70 "skinny" NVFP4 kernel for M<=64
# (decode / speculative-verification widths), env-gated by
# VLLM_SKINNY_NVFP4=1. Weights are kept in the checkpoint's native
# layout (codes [N][K/2], fp8-e4m3 group-16 scales [N][K/16]) alongside
# the marlin-repacked copy, which serves M>64 (prefill).
#
# All M-dependent dispatch lives INSIDE one torch custom op whose eager
# body sees the true runtime M — torch.compile treats the op as opaque,
# so no shape-dependent branch can be constant-folded into a compiled
# graph. Eager-only self-check vs marlin auto-disables on mismatch.
#
# The lm_head section of the upstream patch is intentionally NOT ported:
# this tree carries the native NVFP4 lm_head loader via
# VLLM_SM70_LM_HEAD_NATIVE (vllm/model_executor/kernels/lm_head/sm70.py).

import os

import torch
from torch.library import custom_op

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_SKINNY_ENABLED = os.environ.get("VLLM_SKINNY_NVFP4", "0") == "1"
_SKINNY_SRC = os.environ.get(
    "VLLM_SKINNY_NVFP4_SRC",
    # Default: the kernel source shipped alongside this package. Set
    # VLLM_SKINNY_NVFP4_SRC to override. Never a machine-specific path.
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "kernels", "skinny_kernels.cu"),
)

# Upper M bound for the skinny WMMA path; above it, marlin serves the
# GEMM. Env-tunable for dispatch experiments (e.g., 7 sends the whole
# batch regime to marlin, keeping only the SIMT decode band).
_SKINNY_MAX_M = int(os.environ.get("VLLM_SKINNY_MAX_M", "64"))
# QPN (Volta-native four-quadpair mma.m8n8k4) owns M 4..16 when enabled:
# frontier from qpn_sweep_20260810 — 1.28x M=5 / 1.69x M=8 / 1.29x M=11 /
# 1.22x M=16 vs prior best. Costs one extra weight-sized copy per layer
# (fragment-order prepack, built once at load); disable with
# VLLM_SKINNY_QPN=0 if the memory envelope needs it back.
_QPN_ENABLED = os.environ.get("VLLM_SKINNY_QPN", "1") == "1"
# Drop the checkpoint-native stash once the qpn prepack exists: the
# prepack is a byte-equal permutation, so serving returns to the pre-QPN
# weight footprint (fp32 GDN state then fits). M1-3 route to
# gemm_qpn_simt (qpn-layout SIMT), M17+ to marlin (conceded band).
_QPN_DROP_CT = os.environ.get("VLLM_SKINNY_DROP_CT", "0") == "1"
# QPN2: geometry-winner kernel for M 4..8 (qpn_matrix/qpn_msweep
# 2026-08-17: weighted 637 GB/s vs 441; -30.7% QPN time). Same prepack,
# same bytes — kernel and launch geometry only. VLLM_SKINNY_QPN2=0
# restores the fixed-4-warp kernel.
_QPN2_ENABLED = os.environ.get("VLLM_SKINNY_QPN2", "1") == "1"
# (K, N) -> (splitk, nacc), measured winners; heuristic covers the rest.
# graph-mode winners (qpn_graphmatrix 2026-08-17): captured-replay is
# the serving regime and reshuffles two cells vs the eager matrix.
_QPN2_TABLE = {(1536, 5120): (16, 2), (4352, 5120): (16, 2),
               (5120, 8704): (8, 2), (5120, 4096): (16, 2),
               (5120, 2048): (32, 2), (5120, 62080): (8, 1),
               (5120, 3584): (16, 2)}


def _qpn2_cfg(k: int, n: int):
    cfg = _QPN2_TABLE.get((k, n))
    if cfg is not None and (k // 16) % cfg[0] == 0:
        return cfg
    # smallest split that puts ~640+ warps in flight (80 SMs x 8)
    for s in (8, 16, 32):
        if (k // 16) % s == 0 and (n // 32) * s >= 640:
            return (s, 2 if s >= 16 else 1)
    for s in (16, 8):
        if (k // 16) % s == 0:
            return (s, 2 if s >= 16 else 1)
    return None


_SELF_CHECK_CALLS = 3
_SELF_CHECK_TOL = 3e-2


def _qpn_prepack(codes: torch.Tensor, scales: torch.Tensor):
    """Fragment-order prepack for gemm_qpn: [tile N/32][group K/16]
    [lane 32] x 8B, nibbles pre-interleaved so the TM decoder's (i, i+4)
    output is exactly the adjacent-k B-fragment register pair. Pure
    permutation of the checkpoint bytes."""
    n, k2 = codes.shape
    k = k2 * 2
    if n % 32 or k % 64:
        return None, None
    dev = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                           8, 10, 12, 14, 9, 11, 13, 15], device=dev)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    qc = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=dev)
    qs = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=dev)
    # Chunk the gather: the int64 broadcast-index intermediates are 16x
    # the payload — one-shot on lm_head (37984x5120) spikes 2.4 GiB and
    # OOM'd the memory-profiler forward. Cap transients at ~300 MB.
    chunk = max(1, 36864 // groups)
    for t0 in range(0, tiles, chunk):
        t1 = min(t0 + chunk, tiles)
        tt = t1 - t0
        ncol = (torch.arange(t0, t1, device=dev).view(tt, 1) * 32
                + col.view(1, 32))
        nb = nib[ncol.view(tt, 1, 32, 1).expand(tt, groups, 32, 16),
                 kidx.view(1, groups, 1, 16).expand(tt, groups, 32, 16)]
        qc[t0:t1] = nb[..., 0::2] | (nb[..., 1::2] << 4)
        qs[t0:t1] = scales[ncol.view(tt, 1, 32).expand(tt, groups, 32),
                           g.view(1, groups, 1).expand(tt, groups, 32)]
    del nib
    return qc.view(-1).contiguous(), qs.view(-1).contiguous()


def _qpn_stash(layer) -> None:
    """Attach skinny_qpn_codes/scales (empty tensors when disabled or
    shape-ineligible, so the custom-op signature stays uniform)."""
    qc = qs = None
    if _QPN_ENABLED:
        qc, qs = _qpn_prepack(layer.skinny_codes.data,
                              layer.skinny_scales.data)
    empty = layer.skinny_codes.data.new_empty(0)
    layer.skinny_qpn_codes = torch.nn.Parameter(
        qc if qc is not None else empty, requires_grad=False)
    layer.skinny_qpn_scales = torch.nn.Parameter(
        qs if qs is not None else empty, requires_grad=False)
    if _QPN_DROP_CT and qc is not None:
        layer.skinny_codes.data = empty
        layer.skinny_scales.data = empty


_skinny_ext = None
_skinny_ok = True
_self_checks_done = 0


def _get_skinny_ext():
    global _skinny_ext
    if _skinny_ext is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _skinny_ext = load(
            name="skinny_nvfp4_v11",
            sources=[_SKINNY_SRC],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                               "-gencode=arch=compute_70,code=sm_70"],
            verbose=False,
        )
        logger.info_once("SM70 skinny NVFP4 kernel loaded from %s", _SKINNY_SRC)
    return _skinny_ext


_route_log_seen: set = set()

# Per-(route, M) dispatch counter. The op body runs at capture/compile/
# eager time (replays skip it), so these tally dispatch DECISIONS — proof
# that e.g. every M=16 call routes to qpn and zero to wmma/marlin. When
# VLLM_SKINNY_ROUTE_COUNT_FILE is set, counts are flushed (throttled) to
# "<file>.<pid>" per rank for the campaign to read back.
import collections as _collections
import json as _json

_route_counts: "_collections.Counter" = _collections.Counter()
_ROUTE_COUNT_FILE = os.environ.get("VLLM_SKINNY_ROUTE_COUNT_FILE")
_route_flush_n = [0]


def _bump_route_count(route: str, m: int) -> None:
    _route_counts[(route, int(m))] += 1
    if not _ROUTE_COUNT_FILE:
        return
    _route_flush_n[0] += 1
    if _route_flush_n[0] % 50 != 0:
        return
    try:
        payload = {f"{r}:M{mm}": c for (r, mm), c in _route_counts.items()}
        with open(f"{_ROUTE_COUNT_FILE}.{os.getpid()}", "w") as f:
            _json.dump(payload, f)
    except Exception:
        pass


@custom_op("skinny_nvfp4::linear", mutates_args=())
def _skinny_linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                   qpn_codes: torch.Tensor, qpn_scales: torch.Tensor,
                   marlin_w: torch.Tensor, marlin_s: torch.Tensor,
                   marlin_gs: torch.Tensor, workspace: torch.Tensor,
                   gscale: float, n: int, k: int) -> torch.Tensor:
    m = x.shape[0]
    # Frontier (qpn_sweep_20260810): simt M<=3, qpn M4..16 (Volta-native
    # four-quadpair m8n8k4, prepacked weights), wmma 17..64. With QPN
    # disabled the legacy seam applies: simt through M=7 (seam_bench),
    # wmma above.
    has_ct = codes.numel() > 0
    use_qpn = qpn_codes.numel() > 0 and 4 <= m <= 16 \
        and k % 64 == 0 and n % 32 == 0
    # qpn2 owns M 1..8 (msweep: 550-800 GB/s at M=1 vs qpn1/simt);
    # eligibility mirrors use_qpn's shape checks without its m >= 4.
    qpn2_ok = (_QPN2_ENABLED and qpn_codes.numel() > 0 and 1 <= m <= 8
               and k % 64 == 0 and n % 32 == 0)
    qpn2_cfg = _qpn2_cfg(k, n) if qpn2_ok else None
    use_qpn1 = (not has_ct) and qpn_codes.numel() > 0 and 1 <= m <= 3 \
        and k % 64 == 0 and n % 32 == 0
    use_simt = has_ct and (not use_qpn) and 1 <= m <= 7 \
        and k % 128 == 0 and n % 8 == 0
    use_wmma = has_ct and (not use_qpn) and (not use_simt) \
        and m <= _SKINNY_MAX_M and k % 128 == 0 and n % 64 == 0
    route = "qpn2" if qpn2_cfg is not None else "qpn" if use_qpn \
        else "qpn1" if use_qpn1 \
        else "simt" if use_simt else "wmma" if use_wmma else "marlin"
    # Route map: the op body runs at capture/compile/eager time, so one
    # line per unique (route, M) documents what each graph replays.
    key = (route, m, n, k)
    _bump_route_count(route, m)
    if key not in _route_log_seen:
        _route_log_seen.add(key)
        logger.info("Skinny route map: M=%d N=%d K=%d -> %s", m, n, k, route)
    if _skinny_ok and qpn2_cfg is not None:
        return _get_skinny_ext().gemm_qpn2(
            x, qpn_codes, qpn_scales, gscale, n, qpn2_cfg[0], qpn2_cfg[1])
    if _skinny_ok and use_qpn:
        return _get_skinny_ext().gemm_qpn(x, qpn_codes, qpn_scales, gscale, n)
    if _skinny_ok and use_qpn1:
        return _get_skinny_ext().gemm_qpn_simt(x, qpn_codes, qpn_scales,
                                               gscale, n)
    if _skinny_ok and (use_simt or use_wmma):
        ext = _get_skinny_ext()
        fn = ext.gemm_simt if use_simt else ext.gemm_wmma
        return fn(x, codes, scales, gscale)
    return apply_fp4_marlin_linear(
        input=x, weight=marlin_w, weight_scale=marlin_s,
        weight_global_scale=marlin_gs, workspace=workspace,
        size_n=n, size_k=k, bias=None,
    )


@_skinny_linear.register_fake
def _skinny_linear_fake(x, codes, scales, qpn_codes, qpn_scales, marlin_w,
                        marlin_s, marlin_gs, workspace, gscale, n, k):
    return x.new_empty((x.shape[0], n))


def _eager_context() -> bool:
    """True only when neither compiling/tracing nor capturing a graph."""
    if torch.compiler.is_compiling():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False
    return True


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16), with an optional
    custom skinny kernel for M<=64."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        if _SKINNY_ENABLED:
            # Native CT layout is exactly what the skinny kernel consumes;
            # stash it (and the already-multiplicative global) before the
            # marlin repack rewrites the layer's tensors.
            layer.skinny_codes = torch.nn.Parameter(
                layer.weight.data.clone(), requires_grad=False
            )
            layer.skinny_scales = torch.nn.Parameter(
                layer.weight_scale.data.view(torch.uint8).clone(),
                requires_grad=False,
            )
            layer.skinny_gscale = float(
                layer.weight_global_scale.data.float().item()
            )
            _qpn_stash(layer)
            logger.info_once(
                "SM70 skinny NVFP4 path enabled for M<=%d (QPN %s).",
                _SKINNY_MAX_M, "on" if _QPN_ENABLED else "off"
            )
        prepare_fp4_layer_for_marlin(layer)

    def _marlin_apply(self, layer, x, bias):
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _skinny_ok, _self_checks_done
        if not (_SKINNY_ENABLED and hasattr(layer, "skinny_codes")
                and x.dtype == torch.half):
            return self._marlin_apply(layer, x, bias)

        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        xc = x.reshape(-1, k).contiguous()
        y = torch.ops.skinny_nvfp4.linear(
            xc, layer.skinny_codes, layer.skinny_scales,
            layer.skinny_qpn_codes, layer.skinny_qpn_scales,
            layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, layer.skinny_gscale, n, k,
        )
        if bias is not None:
            y = y + bias

        if _self_checks_done < _SELF_CHECK_CALLS and _eager_context() \
                and _skinny_ok and xc.shape[0] <= _SKINNY_MAX_M:
            _self_checks_done += 1
            y_ref = self._marlin_apply(layer, xc, bias)
            denom = y_ref.float().abs().max().clamp(min=1e-6)
            rel = ((y.float() - y_ref.float()).abs().max() / denom).item()
            if rel > _SELF_CHECK_TOL:
                _skinny_ok = False
                logger.error(
                    "Skinny NVFP4 self-check FAILED (rel err %.3e, M=%d, "
                    "N=%d, K=%d); falling back to marlin permanently.",
                    rel, xc.shape[0], n, k,
                )
                return y_ref
            logger.info(
                "Skinny NVFP4 self-check ok (rel err %.3e, M=%d, N=%d, K=%d).",
                rel, xc.shape[0], n, k,
            )
        return y.reshape(x.shape[:-1] + (n,))

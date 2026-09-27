# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import os

from safetensors import safe_open
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

import vllm.envs as envs
from vllm.config.execution_policy import graph_policy, layer_policy
from vllm.config.sm70_dflash2 import (
    capture_sm70_dflash2_config,
    resolved_sm70_dflash2_config,
)
from vllm.config.sm70_runtime import capture_runtime_trace
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.kernels.lm_head import sm70 as lm_head_provider
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

DEFAULT_VOCAB_PADDING_SIZE = 64
logger = init_logger(__name__)


def _new_sm70_lm_head_state(layer):
    prefix = getattr(layer, "prefix", "")
    shard = getattr(layer, "shard_indices", None)
    return lm_head_provider.Sm70LMHeadState(
        layer.weight,
        lm_head_provider.VocabShard(
            getattr(shard, "org_vocab_start_index", 0),
            getattr(shard, "num_org_vocab_padding", 0),
        ),
        is_lm_head=(bool(prefix) and prefix.rsplit(".", 1)[-1] == "lm_head")
        or layer.__class__.__name__ == "ParallelLMHead",
        policy=layer_policy(),
        dflash=resolved_sm70_dflash2_config(),
        trace=capture_runtime_trace(),
    )


def maybe_prepare_sm70_lm_head_top1(layer: torch.nn.Module) -> bool:
    state = getattr(layer, "_sm70_lm_head_state", None)
    if state is None or state.weight is not layer.weight:
        state = _new_sm70_lm_head_state(layer)
        layer._sm70_lm_head_state = state
    prepared = lm_head_provider.maybe_prepare_sm70_lm_head_top1(state)

    # Native NVFP4 lm_head: load the checkpoint's own packed codes when
    # available. Skips tied heads (weight aliased from embed_tokens has
    # no separate checkpoint tensors) and checkpoints without native
    # head keys — the fp16 paths above remain the fallback either way.
    if (
        prepared
        and envs.VLLM_SM70_LM_HEAD_NATIVE
        and not getattr(layer, "_sm70_native_lm_ready", False)
        and type(layer).__name__ == "ParallelLMHead"
        and not getattr(layer, "_sm70_tied_weights", False)
    ):
        try:
            from vllm.config import get_current_vllm_config

            model_dir = get_current_vllm_config().model_config.model
        except Exception:
            model_dir = None
        if model_dir and os.path.isdir(model_dir):
            _load_native_lm_head_codes(layer, model_dir)
    return prepared


def _prepared_sm70_lm_head_state(layer):
    state = getattr(layer, "_sm70_lm_head_state", None)
    return state if state is not None and state.weight is layer.weight else None


def _maybe_sm70_lm_head_forward(layer, x, bias=None):
    state = _prepared_sm70_lm_head_state(layer)
    if state is None:
        return None
    return lm_head_provider._maybe_sm70_lm_head_forward(state, x, bias)


def _maybe_sm70_lm_head_top1(layer, x, bias=None):
    # Native fused path first: consumes the checkpoint's own packed codes
    # with no logits materialization. Falls through to the fp16 paths on
    # any miss (flag off, buffers absent, op missing, shape mismatch).
    native = _maybe_sm70_native_lm_head_top1(layer, x)
    if native is not None:
        return native

    state = _prepared_sm70_lm_head_state(layer)
    if state is None:
        return None
    return lm_head_provider._maybe_sm70_lm_head_top1(state, x, bias)


def _maybe_sm70_dflash2_qpn8_rerank(layer, x, selector_k, bias=None):
    state = _prepared_sm70_lm_head_state(layer)
    if state is None:
        return None
    result = lm_head_provider._maybe_sm70_dflash2_qpn8_rerank(
        state, x, selector_k, bias
    )
    if isinstance(result, lm_head_provider.RerankShadow):
        dense = layer.quant_method.apply(layer, result.x, bias=None)
        return lm_head_provider.complete_shadow(state, result, dense)
    return result


_maybe_sm70_dflash2_lm_head_top20 = _maybe_sm70_dflash2_qpn8_rerank


def _is_sm70_lm_head_fastpath_eligible(layer):
    return lm_head_provider._is_sm70_lm_head_fastpath_eligible(
        _new_sm70_lm_head_state(layer)
    )


def _is_sm70_dflash2_qpn8_rerank_eligible(layer):
    return lm_head_provider._is_sm70_dflash2_qpn8_rerank_eligible(
        _new_sm70_lm_head_state(layer)
    )


def _sm70_lm_head_top1_default():
    return not graph_policy().compile_graph


def _load_native_lm_head_codes(
    layer: torch.nn.Module,
    model_dir: str,
) -> bool:
    """Load native NVFP4 lm_head codes/scales from the served checkpoint.

    A modelopt NVFP4 checkpoint ships the lm_head already quantized:
    ``lm_head.weight`` [V, K/2] uint8 (two e2m1 codes per byte, low nibble
    first), ``lm_head.weight_scale`` [V, K/16] float8_e4m3, and a scalar
    ``lm_head.weight_scale_2`` global scale. Loading them verbatim skips
    both the fp16 weight copy (2.4 GiB -> 0.6 GiB for a 248k vocab) and
    any requantization pass.

    Loads this rank's vocab shard only: rows [rank*n, (rank+1)*n) of the
    checkpoint tensors, where n = num_embeddings_per_partition. Falls
    back cleanly (returns False) when the checkpoint has no native head,
    the shapes mismatch (padded vocab), or the shard range exceeds the
    source rows — the caller then keeps the fp16 path.

    Sets on success:
      layer._sm70_native_lm_codes   uint8 [n, K/2]
      layer._sm70_native_lm_scales  uint8 view of fp8 [n, K/16]
      layer._sm70_native_lm_gscale  float
      layer._sm70_native_lm_ready   True
    """
    import glob
    import json
    import os

    try:
        rank = get_tensor_model_parallel_rank()
    except Exception:
        return False

    weight = getattr(layer, "weight", None)
    if weight is None or weight.dim() != 2:
        return False
    n, k = weight.shape
    if n < 1 or k % 128 != 0:
        return False

    # Locate the shard holding the native head via the index, else scan.
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    shard: str | None = None
    if os.path.exists(idx_path):
        try:
            with open(idx_path) as f:
                wm = json.load(f)["weight_map"]
            shard = wm.get("lm_head.weight")
        except Exception:
            shard = None
    if shard is None:
        for cand in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            try:
                with safe_open(cand, framework="pt") as f:
                    if "lm_head.weight" in f.keys():
                        shard = cand
                        break
            except Exception:
                continue
    if shard is None:
        return False

    path = os.path.join(model_dir, shard)
    try:
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            if "lm_head.weight" not in keys or "lm_head.weight_scale" not in keys:
                return False
            total = f.get_slice("lm_head.weight").get_shape()[0]
            off = rank * n
            if off + n > total:
                logger.warning_once(
                    "Native lm_head shard [%d:%d] exceeds source rows %d "
                    "(padded vocab?); keeping fp16 head.",
                    off, off + n, total)
                return False
            codes = f.get_tensor("lm_head.weight")[off:off + n]
            scales = f.get_tensor("lm_head.weight_scale")[off:off + n]
            g2 = f.get_tensor("lm_head.weight_scale_2")
    except Exception:
        logger.warning_once("Native lm_head read failed; keeping fp16 head.")
        return False

    dev = weight.device
    layer._sm70_native_lm_codes = codes.to(torch.uint8).to(dev).contiguous()
    layer._sm70_native_lm_scales = (
        scales.view(torch.uint8).to(dev).contiguous())
    layer._sm70_native_lm_gscale = float(g2.float().item())
    layer._sm70_native_lm_ready = True
    logger.info_once(
        "Native NVFP4 lm_head loaded: rows %d..%d of %d, %.0f MiB packed "
        "(gscale %.4e).", off, off + n, total,
        layer._sm70_native_lm_codes.numel() / 2**20,
        layer._sm70_native_lm_gscale)
    return True


def _maybe_sm70_native_lm_head_top1(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused NVFP4 GEMM+argmax over the checkpoint's own packed codes.

    Consumes the buffers _load_native_lm_head_codes stashed on the layer.
    The kernel returns one (value, index) pair per block; a tiny final
    max+argmax over those pairs yields the global (value, index). Index
    is LOCAL to this rank's shard — the caller's contract (matching the
    sm70_f16_lm_head_top1_out call sites) is the global vocab index, so
    add org_vocab_start_index here.
    """
    if not envs.VLLM_SM70_LM_HEAD_NATIVE:
        return None
    if not getattr(layer, "_sm70_native_lm_ready", False):
        return None
    if x.dim() < 2 or x.shape[-2] != 1:
        return None
    if x.dtype != torch.float16 or not x.is_cuda:
        return None
    if not hasattr(torch.ops._C, "sm70_nvfp4_simt_argmax_out"):
        return None

    codes = layer._sm70_native_lm_codes
    scales = layer._sm70_native_lm_scales
    gscale = layer._sm70_native_lm_gscale
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    bvals, bidxs = torch.ops._C.sm70_nvfp4_simt_argmax_out(
        x_2d, codes, scales, float(gscale))
    mx = bvals.max()
    blk = (bvals == mx).int().argmax()
    idx = (bidxs[blk].to(torch.long)
           + layer.shard_indices.org_vocab_start_index)
    return mx.view(1), idx.view(1)


def __getattr__(name):
    # Stateless helper import compatibility. Provider never imports this module.
    if name.startswith("_SM70_") or name.startswith("_sm70_dflash2_"):
        return getattr(lm_head_provider, name)
    raise AttributeError(name)


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
            return

        maybe_prepare_sm70_lm_head_top1(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sm70_out = _maybe_sm70_lm_head_forward(layer, x, bias)
        if sm70_out is not None:
            return sm70_out
        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# --8<-- [start:vocab_parallel_embedding]
@PluggableLayer.register("vocab_parallel_embedding")
class VocabParallelEmbedding(PluggableLayer):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1025 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1010 | ... | 1025 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  528 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  | 528 | ... |  543 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config for the layer
        prefix: full name of the layer in the state dict
    """  # noqa: E501

    # --8<-- [end:vocab_parallel_embedding]

    def __setattr__(self, name, value):
        if name == "weight":
            state = self.__dict__.get("_modules", {}).get("_sm70_lm_head_state")
            if state is not None and state.weight is not value:
                del self._sm70_lm_head_state
        super().__setattr__(name, value)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith(("_sm70_f16_", "_sm70_dflash2_")):
                state = self._modules.get("_sm70_lm_head_state")
                if state is not None:
                    return getattr(state, name)
            raise

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        quant_method: QuantizeMethodBase | None = None,
    ):
        super().__init__()
        self._sm70_dflash2_policy = capture_sm70_dflash2_config()
        self.prefix = prefix

        # Keep the input dimensions.
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        # Model-specific embeddings can preselect a storage method. This is
        # required by Qwen4Exp PLE, whose table remains FP8 even when the
        # routed experts use a different checkpoint quantization config.
        if quant_method is None and quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocabulary dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.

        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        if self.tp_size < 2:
            return None

        base_embeddings: list[int] = []
        added_embeddings: list[int] = []
        padding: list[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        # If the parameter is a gguf weight, then load it directly.
        if getattr(param, "is_gguf_weight_type", None):
            param.data.copy_(loaded_weight)
            param.weight_type = loaded_weight.item()
            return
        elif isinstance(param, UninitializedParameter):
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = self.num_embeddings_per_partition
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            if (
                loaded_weight.ndim == 0
                and param.data.ndim == 1
                and param.data.numel() == 1
            ):
                loaded_weight = loaded_weight.reshape(1)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BasevLLMParameter)
                else param.pack_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == self.org_vocab_size

        # Copy the data. Select chunk corresponding to current shard.
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            if output_parallel.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                # Each token row has exactly one TP owner. Communicate the raw
                # FP8 bytes as int8 because NCCL does not reduce FP8 directly.
                comm_output = output_parallel.view(torch.int8)
                comm_output.masked_fill_(input_mask.unsqueeze(-1), 0)
                output = tensor_model_parallel_all_reduce(comm_output)
                return output.view(output_parallel.dtype)
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    def maybe_get_sm70_lm_head_top1(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return _maybe_sm70_lm_head_top1(self, hidden_states, bias)

    def maybe_get_sm70_dflash2_top20(
        self,
        hidden_states: torch.Tensor,
        selector_k: int,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return _maybe_sm70_dflash2_lm_head_top20(self, hidden_states, selector_k, bias)

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# --8<-- [start:parallel_lm_head]
@PluggableLayer.register("parallel_lm_head")
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    # --8<-- [end:parallel_lm_head]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            if hasattr(self, "_sm70_lm_head_state"):
                del self._sm70_lm_head_state
            self.weight = embed_tokens.weight
            return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")

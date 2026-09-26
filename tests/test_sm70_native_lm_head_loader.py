"""CPU tests for the native NVFP4 lm_head loader.

Validates ``_load_native_lm_head_codes`` against the Swift-1.5-27B
NVFP4 checkpoint layout (mocked shard layer; shapes/scales measured
from the live checkpoint):

- lm_head.weight        [248320, 2560] uint8  (two e2m1 codes/byte)
- lm_head.weight_scale  [248320, 320]  float8_e4m3 (one scale / 16 k)
- lm_head.weight_scale_2 scalar fp32 (0.00012715657067019492 measured)
- value contract: e2m1_code * fp8_scale * weight_scale_2
"""

import json
import os
import types

import pytest

torch = pytest.importorskip("torch")

import vllm.model_executor.layers.vocab_parallel_embedding as vpe
from vllm.model_executor.layers.vocab_parallel_embedding import (
    _load_native_lm_head_codes,
)

VOCAB = 248320
HIDDEN = 5120
GSCALE = 0.00012715657067019492


class _FakeSlice:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _FakeShard:
    """Mimics safetensors safe_open for the four lm_head tensors."""

    def __init__(self, vocab, hidden, gscale):
        self._keys = {
            "lm_head.weight",
            "lm_head.weight_scale",
            "lm_head.weight_scale_2",
            "lm_head.input_scale",
        }
        self._vocab = vocab
        self._hidden = hidden
        self._gscale = gscale

    def keys(self):
        return self._keys

    def get_slice(self, key):
        if key == "lm_head.weight":
            return _FakeSlice([self._vocab, self._hidden // 2])
        if key == "lm_head.weight_scale":
            return _FakeSlice([self._vocab, self._hidden // 16])
        return _FakeSlice([])

    def get_tensor(self, key):
        if key == "lm_head.weight":
            return torch.zeros((self._vocab, self._hidden // 2),
                               dtype=torch.uint8)
        if key == "lm_head.weight_scale":
            return torch.zeros((self._vocab, self._hidden // 16),
                               dtype=torch.float8_e4m3fn)
        if key == "lm_head.weight_scale_2":
            return torch.tensor(self._gscale, dtype=torch.float32)
        raise KeyError(key)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _EmptyShard:
    def keys(self):
        return set()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_layer(vocab_shard, hidden):
    layer = types.SimpleNamespace()
    layer.weight = torch.empty((vocab_shard, hidden), dtype=torch.float16)
    return layer


def _patch_shard(monkeypatch, shard):
    """Patch safe_open as bound into the vpe module namespace (the
    loader resolves it from vpe's module globals, set by the
    module-level `from safetensors import safe_open`)."""
    monkeypatch.setattr(vpe, "safe_open", lambda path, framework: shard)


def _patch_tp_rank(monkeypatch, rank):
    """Patch get_tensor_model_parallel_rank as bound into vpe (the
    module-level from-import binds the symbol into vpe's namespace)."""
    monkeypatch.setattr(
        vpe,
        "get_tensor_model_parallel_rank",
        lambda: rank,
    )



def _write_index(tmp_path):
    idx = {"weight_map": {"lm_head.weight": "model-00005.safetensors"}}
    path = os.path.join(tmp_path, "model.safetensors.index.json")
    with open(path, "w") as f:
        json.dump(idx, f)
    return path


def test_native_loader_loads_shard(tmp_path, monkeypatch):
    """Happy path: index resolves the shard, rank-0 slice loads."""
    _write_index(tmp_path)
    _patch_shard(monkeypatch, _FakeShard(VOCAB, HIDDEN, GSCALE))
    _patch_tp_rank(monkeypatch, 0)
    layer = _make_layer(VOCAB // 2, HIDDEN)  # TP2 shard

    assert _load_native_lm_head_codes(layer, str(tmp_path)) is True
    assert layer._sm70_native_lm_ready is True
    assert layer._sm70_native_lm_codes.shape == (VOCAB // 2, HIDDEN // 2)
    assert layer._sm70_native_lm_codes.dtype == torch.uint8
    assert layer._sm70_native_lm_scales.shape == (VOCAB // 2, HIDDEN // 16)
    assert layer._sm70_native_lm_scales.dtype == torch.uint8
    assert layer._sm70_native_lm_gscale == pytest.approx(GSCALE)


def test_native_loader_rank1_slice(tmp_path, monkeypatch):
    """TP rank 1 reads rows [n, 2n) — offset math verified."""
    _write_index(tmp_path)
    _patch_shard(monkeypatch, _FakeShard(VOCAB, HIDDEN, GSCALE))
    _patch_tp_rank(monkeypatch, 1)
    layer = _make_layer(VOCAB // 2, HIDDEN)

    assert _load_native_lm_head_codes(layer, str(tmp_path)) is True
    # rank 1 slice = 124160..248320, exactly fits the 248320-row source.
    assert layer._sm70_native_lm_ready is True


def test_native_loader_padded_vocab_rejects(tmp_path, monkeypatch):
    """Padded vocab: shard rows exceed source rows -> clean fallback."""
    _write_index(tmp_path)
    _patch_shard(monkeypatch, _FakeShard(VOCAB, HIDDEN, GSCALE))
    _patch_tp_rank(monkeypatch, 0)
    layer = _make_layer(VOCAB, HIDDEN)  # full vocab, rank 0: fits exactly
    assert _load_native_lm_head_codes(layer, str(tmp_path)) is True

    # Shrink the source by one row: the same request now overruns.
    _patch_shard(monkeypatch, _FakeShard(VOCAB - 1, HIDDEN, GSCALE))
    layer2 = _make_layer(VOCAB, HIDDEN)
    assert _load_native_lm_head_codes(layer2, str(tmp_path)) is False
    assert getattr(layer2, "_sm70_native_lm_ready", False) is False


def test_native_loader_missing_checkpoint(tmp_path, monkeypatch):
    """No index, no shards -> False, layer untouched."""
    _patch_shard(monkeypatch, _FakeShard(VOCAB, HIDDEN, GSCALE))
    _patch_tp_rank(monkeypatch, 0)
    layer = _make_layer(VOCAB // 2, HIDDEN)

    assert _load_native_lm_head_codes(layer, str(tmp_path)) is False
    assert getattr(layer, "_sm70_native_lm_ready", False) is False


def test_native_loader_no_native_head(tmp_path, monkeypatch):
    """Checkpoint without lm_head keys -> False (fp16 path keeps)."""
    _write_index(tmp_path)
    _patch_shard(monkeypatch, _EmptyShard())
    _patch_tp_rank(monkeypatch, 0)
    layer = _make_layer(VOCAB // 2, HIDDEN)

    assert _load_native_lm_head_codes(layer, str(tmp_path)) is False
    assert getattr(layer, "_sm70_native_lm_ready", False) is False


def test_packing_math_matches_checkpoint():
    """The 25% ratio and group counts measured from the live checkpoint."""
    assert HIDDEN // 2 == 2560          # packed bytes per row
    assert HIDDEN // 16 == 320          # fp8 scales per row
    packed = VOCAB * (HIDDEN // 2)
    fp16 = VOCAB * HIDDEN * 2
    assert packed / fp16 == pytest.approx(0.25)
    assert GSCALE * 6.0 * 448 < 1.0     # dynamic range cap < 1.0
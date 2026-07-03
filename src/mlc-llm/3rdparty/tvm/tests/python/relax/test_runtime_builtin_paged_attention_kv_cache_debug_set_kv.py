# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Parity test for DebugSetKV / DebugGetKV round-trip.

Verifies the M1 lynchpin of the PD-split work: externally-computed K/V can be
injected into a paged KV cache for a sequence whose pages are reserved via
BeginForward, and read back bit-for-bit identical.

The "external prefill producer" is simulated by running a real prefill on a
source sequence and extracting its K/V via DebugGetKV; that K/V is then injected
into a fresh destination sequence via DebugSetKV, and DebugGetKV is read back
on the destination for comparison.

CPU-only by design: this isolates the DebugSetKV C++ logic from Metal/backend
issues. Metal parity is exercised separately by the engine-level tests.
"""
import itertools

import numpy as np
import pytest
import tvm
import tvm_ffi
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _attention_prefill_cpu,
    _attention_decode_cpu,
    _attention_prefill_ragged_cpu,
    _merge_state_inplace_cpu,
    _copy_single_page_cpu,
    _compact_kv_copy_cpu,
    llama_rope_with_position_map,
    tree_attn_cpu,
    tree_attn_with_paged_kv_cache_cpu,
)
from tvm.s_tir import dlight as dl

# ----------------------------- config -----------------------------
reserved_nseq = 4
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
rope_scale = 1.0
rope_theta = 1e4
rope_scaling = {}
device = tvm.cpu()
target = tvm.target.Target.from_device(device)


def _build_kernels(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype):
    """Build all kernels the paged KV cache constructor requires (CPU variants)."""
    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _attention_prefill_cpu(num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling),
        _attention_decode_cpu(num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling),
        _attention_prefill_cpu(num_kv_heads, num_qo_heads, head_dim, dtype, True, rope_scaling),
        _attention_decode_cpu(num_kv_heads, num_qo_heads, head_dim, dtype, True, rope_scaling),
        _attention_prefill_ragged_cpu(
            num_kv_heads, num_qo_heads, head_dim, head_dim, dtype, rope_scaling
        ),
        tree_attn_cpu(num_kv_heads, num_qo_heads, head_dim, dtype, rope_scaling),
        tree_attn_with_paged_kv_cache_cpu(
            num_kv_heads, num_qo_heads, head_dim, dtype, rope_scaling
        ),
        _merge_state_inplace_cpu(dtype),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype, rope_scaling
        ),
        _copy_single_page_cpu(num_kv_heads, page_size, head_dim, dtype),
        _compact_kv_copy_cpu(num_kv_heads, head_dim, dtype),
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        builts.append(tvm.tirx.build(mod["main"], target=target).main)
    return builts


def _create_kv_cache(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype, builts):
    (
        ftranspose_append,
        fcopy_cache,
        fattn_prefill,
        fattn_decode,
        fattn_prefill_sliding_window,
        fattn_decode_sliding_window,
        fattn_prefill_ragged,
        fattn_prefill_with_tree_mask,
        fattn_prefill_with_tree_mask_paged_kv_cache,
        fmerge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcompact_copy,
    ) = builts
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    return fcreate(
        tvm_ffi.Shape(
            [reserved_nseq, maximum_total_seq_length, prefill_chunk_size, page_size, 0]
        ),
        tvm_ffi.Shape([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        tvm_ffi.Shape([int(AttnKind.MHA) for _ in range(num_layers)]),
        False,
        RopeMode.NONE,
        rope_scale,
        rope_theta,
        None,
        tvm.runtime.empty((), dtype, device=device),
        ftranspose_append,
        None,
        ["tirx", fattn_prefill_ragged],
        ["tirx", fattn_prefill],
        ["tirx", fattn_decode],
        ["tirx", fattn_prefill_sliding_window],
        ["tirx", fattn_decode_sliding_window],
        ["tirx", fattn_prefill_with_tree_mask_paged_kv_cache],
        ["tirx", fattn_prefill_with_tree_mask],
        [],
        [fmerge_state],
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    )


@pytest.fixture(
    params=itertools.product(
        [1, 2, 4],          # num_layers: covers single-layer and multi-layer CreateView paths
        [40, 33, 16],       # length: multi-page, non-page-aligned, single-page
        ["float32", "float16"],
    )
)
def cfg(request):
    num_layers, length, dtype = request.param
    num_qo_heads = 8
    num_kv_heads = 4
    head_dim = 64
    return {
        "num_layers": num_layers,
        "length": length,
        "dtype": dtype,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "sm_scale": head_dim ** (-0.5),
    }


def test_debug_set_kv_round_trip(cfg):
    """prefill(seq_a) -> GetKV(seq_a) -> reserve(seq_b) -> SetKV(seq_b) -> GetKV(seq_b) == bit-for-bit."""
    num_layers = cfg["num_layers"]
    length = cfg["length"]
    dtype = cfg["dtype"]
    num_qo_heads = cfg["num_qo_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    sm_scale = cfg["sm_scale"]

    builts = _build_kernels(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype)
    cache = _create_kv_cache(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype, builts)

    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fattention = tvm.get_global_func("vm.builtin.attention_kv_cache_attention_with_fused_qkv")
    fset_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_set_kv")
    fget_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    # --- run a real prefill on seq_a to produce "external" K/V ---
    seq_a = 0
    fadd_sequence(cache, seq_a)
    fbegin_forward(cache, tvm_ffi.Shape([seq_a]), tvm_ffi.Shape([length]))
    rng = np.random.RandomState(0)
    total_qkv_heads = num_qo_heads + 2 * num_kv_heads
    for layer_id in range(num_layers):
        qkv_np = rng.randn(length, total_qkv_heads, head_dim).astype(dtype)
        qkv = tvm.runtime.tensor(qkv_np, device=device)
        out = tvm.runtime.empty((length, num_qo_heads, head_dim), dtype, device=device)
        fattention(cache, layer_id, sm_scale, qkv, out)
    fend_forward(cache)

    # --- extract K/V from seq_a (the "ground truth" from the external producer) ---
    k_ref = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    v_ref = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    fget_kv(cache, seq_a, 0, length, k_ref, v_ref)
    k_ref_np = k_ref.numpy()
    v_ref_np = v_ref.numpy()

    # --- reserve pages on a fresh seq_b, then inject the same K/V ---
    seq_b = 1
    fadd_sequence(cache, seq_b)
    fbegin_forward(cache, tvm_ffi.Shape([seq_b]), tvm_ffi.Shape([length]))
    fend_forward(cache)
    fset_kv(cache, seq_b, 0, k_ref, v_ref)

    # --- read back from seq_b and require bit-for-bit equality ---
    k_out = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    v_out = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    fget_kv(cache, seq_b, 0, length, k_out, v_out)

    np.testing.assert_array_equal(k_out.numpy(), k_ref_np)
    np.testing.assert_array_equal(v_out.numpy(), v_ref_np)


def test_debug_set_kv_rejects_cache_dtype_mismatch():
    """DebugSetKV should fail loudly when external K/V dtype differs from the cache dtype."""
    num_layers = 2
    length = 16
    cache_dtype = "float16"
    kv_dtype = "float32"
    num_qo_heads = 8
    num_kv_heads = 4
    head_dim = 64

    builts = _build_kernels(num_layers, num_qo_heads, num_kv_heads, head_dim, cache_dtype)
    cache = _create_kv_cache(
        num_layers, num_qo_heads, num_kv_heads, head_dim, cache_dtype, builts
    )

    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fset_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_set_kv")

    seq_id = 0
    fadd_sequence(cache, seq_id)
    fbegin_forward(cache, tvm_ffi.Shape([seq_id]), tvm_ffi.Shape([length]))
    fend_forward(cache)

    k_bad = tvm.runtime.empty(
        (num_layers, length, num_kv_heads, head_dim), kv_dtype, device=device
    )
    v_bad = tvm.runtime.empty(
        (num_layers, length, num_kv_heads, head_dim), kv_dtype, device=device
    )

    with pytest.raises(Exception, match="DebugSetKV expects K/V dtype to match the cache dtype"):
        fset_kv(cache, seq_id, 0, k_bad, v_bad)


def test_debug_set_kv_chunked_start_pos(cfg):
    """Inject K/V in two chunks: [0, s) at start_pos=0, then [s, L) at start_pos=s>0.

    Exercises DebugSetKV's start_pos>0 position-map windowing (and, for the non-page-aligned
    lengths, a chunk boundary that crosses pages). The recombined cache must equal the source
    bit-for-bit.
    """
    num_layers = cfg["num_layers"]
    length = cfg["length"]
    dtype = cfg["dtype"]
    num_qo_heads = cfg["num_qo_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    sm_scale = cfg["sm_scale"]
    s = length // 2
    assert 0 < s < length  # both chunks non-empty

    builts = _build_kernels(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype)
    cache = _create_kv_cache(num_layers, num_qo_heads, num_kv_heads, head_dim, dtype, builts)

    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fattention = tvm.get_global_func("vm.builtin.attention_kv_cache_attention_with_fused_qkv")
    fset_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_set_kv")
    fget_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    # produce reference K/V via a real prefill on seq_a
    seq_a = 0
    fadd_sequence(cache, seq_a)
    fbegin_forward(cache, tvm_ffi.Shape([seq_a]), tvm_ffi.Shape([length]))
    rng = np.random.RandomState(1)
    total_qkv_heads = num_qo_heads + 2 * num_kv_heads
    for layer_id in range(num_layers):
        qkv_np = rng.randn(length, total_qkv_heads, head_dim).astype(dtype)
        qkv = tvm.runtime.tensor(qkv_np, device=device)
        out = tvm.runtime.empty((length, num_qo_heads, head_dim), dtype, device=device)
        fattention(cache, layer_id, sm_scale, qkv, out)
    fend_forward(cache)
    k_ref = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    v_ref = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    fget_kv(cache, seq_a, 0, length, k_ref, v_ref)
    k_ref_np, v_ref_np = k_ref.numpy(), v_ref.numpy()

    # reserve seq_b for the full length, then inject in two chunks (the 2nd at start_pos=s>0)
    seq_b = 1
    fadd_sequence(cache, seq_b)
    fbegin_forward(cache, tvm_ffi.Shape([seq_b]), tvm_ffi.Shape([length]))
    fend_forward(cache)

    def _chunk(arr, lo, hi):
        return tvm.runtime.tensor(np.ascontiguousarray(arr[:, lo:hi]), device=device)

    fset_kv(cache, seq_b, 0, _chunk(k_ref_np, 0, s), _chunk(v_ref_np, 0, s))
    fset_kv(cache, seq_b, s, _chunk(k_ref_np, s, length), _chunk(v_ref_np, s, length))

    k_out = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    v_out = tvm.runtime.empty((num_layers, length, num_kv_heads, head_dim), dtype, device=device)
    fget_kv(cache, seq_b, 0, length, k_out, v_out)

    np.testing.assert_array_equal(k_out.numpy(), k_ref_np)
    np.testing.assert_array_equal(v_out.numpy(), v_ref_np)

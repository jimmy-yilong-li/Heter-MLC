"""Engine-level PD external-KV smoke test for Qwen3-0.6B.

This keeps the producer side intentionally outside the serving engine:

1. Run a normal VM prefill only as a stub producer for reference K/V and a seed token.
2. Submit ExternalKVData(K, V, prefill_length, [seed_token]) to SyncMLCEngine.
3. Let the engine action reserve pages, call DebugSetKV, and let BatchDecode emit token(s).

The pass criterion is exact token-list parity against a VM reference continuation plus correct
emitted-token accounting. This is an engine-action wiring smoke, not a CoreML/ANE producer
correctness proof. The engine request path does not call low-level reserve/debug_set_kv from
Python.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import tvm
from huggingface_hub import snapshot_download
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime.vm import VirtualMachine
from tvm_ffi import Shape

from mlc_llm.interface.jit import jit
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import EngineConfig, ExternalKVData
from mlc_llm.serve.sync_engine import SyncMLCEngine

MODEL = Path(os.environ.get("QWEN3_MLC") or snapshot_download("mlc-ai/Qwen3-0.6B-q4f16_1-MLC"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "metal"
device = tvm.device(DEV, 0)
print(f"=== device: {DEV} ===")

cfg = json.load(open(MODEL / "mlc-chat-config.json", encoding="utf-8"))
mc = cfg["model_config"]
num_layers = mc["num_hidden_layers"]
num_kv_heads = mc["num_key_value_heads"]
head_dim = mc["head_dim"]
kv_dtype = "float16"
prompt_ids = [13 + i * 500 for i in range(24)]
prefill_length = len(prompt_ids)
max_tokens = int(os.environ.get("PD_MAX_TOKENS", "2"))

print(
    f"model={MODEL} L={num_layers} Hkv={num_kv_heads} D={head_dim} "
    f"KV dtype={kv_dtype} prompt_len={prefill_length}"
)

print("=== JIT compiling/loading VM reference ===")
jit_result = jit(MODEL, {}, DEV)
lib_path = jit_result.model_lib_path
print("lib:", lib_path)

executable = tvm.runtime.load_module(lib_path)
vm = relax.VirtualMachine(executable, device)
mod = vm.module
metadata = json.loads(VirtualMachine(executable, tvm.cpu())["_metadata"]())
params_map, _ = tvmjs.load_tensor_cache(str(MODEL), device)
params = [params_map[p["name"]] for p in metadata["params"]]

embed = mod["embed"]
prefill = mod["prefill"]
decode = mod["decode"]
create_kv = mod["create_tir_paged_kv_cache"]
add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
begin_fwd = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
end_fwd = tvm.get_global_func("vm.builtin.kv_state_end_forward")
get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")
reshape = tvm.get_global_func("vm.builtin.reshape")


def embed_tokens(token_ids):
    token_tensor = tvm.runtime.tensor(np.array(token_ids).astype("int32"), device=device)
    emb = embed(token_tensor, params)
    return reshape(emb, Shape([1, emb.shape[0], emb.shape[1]]))


def make_kv():
    return create_kv(
        Shape([2]),
        Shape([2048]),
        Shape([cfg["prefill_chunk_size"]]),
        Shape([16]),
        Shape([int(cfg["sliding_window_size"] != -1)]),
    )


print("=== VM producer/reference ===")
kv = make_kv()
add_seq(kv, 0)
begin_fwd(kv, Shape([0]), Shape([prefill_length]))
logits_prefill, kv = prefill(embed_tokens(prompt_ids), kv, params)
end_fwd(kv)
seed_token = int(np.asarray(logits_prefill.numpy()).reshape(-1).argmax())
print(f"VM normal prefill seed token={seed_token}")

k_data = tvm.runtime.empty(
    (num_layers, prefill_length, num_kv_heads, head_dim), kv_dtype, device=device
)
v_data = tvm.runtime.empty(
    (num_layers, prefill_length, num_kv_heads, head_dim), kv_dtype, device=device
)
get_kv(kv, 0, 0, prefill_length, k_data, v_data)
print(f"producer K{k_data.shape} V{v_data.shape}")

begin_fwd(kv, Shape([0]), Shape([1]))
logits_decode, kv = decode(embed_tokens([seed_token]), kv, params)
end_fwd(kv)
reference_tokens = [int(np.asarray(logits_decode.numpy()).reshape(-1).argmax())]
while len(reference_tokens) < max_tokens:
    begin_fwd(kv, Shape([0]), Shape([1]))
    logits_decode, kv = decode(embed_tokens([reference_tokens[-1]]), kv, params)
    end_fwd(kv)
    reference_tokens.append(int(np.asarray(logits_decode.numpy()).reshape(-1).argmax()))
print(f"VM reference continuation tokens={reference_tokens}")

print("=== SyncMLCEngine ExternalKVData request ===")
emitted_token_ids = []
finish_reasons = []


def request_stream_callback(delta_outputs):
    for delta_output in delta_outputs:
        request_id, stream_outputs = delta_output.unpack()
        for stream_output in stream_outputs:
            emitted_token_ids.extend(stream_output.delta_token_ids)
            if stream_output.finish_reason is not None:
                finish_reasons.append(str(stream_output.finish_reason))
            print(
                f"callback request={request_id} delta={stream_output.delta_token_ids} "
                f"finish={stream_output.finish_reason}"
            )


engine_config = EngineConfig(
    max_num_sequence=1,
    max_total_sequence_length=2048,
    max_single_sequence_length=2048,
    prefill_chunk_size=cfg["prefill_chunk_size"],
    prefix_cache_mode="disable",
    verbose=False,
)
engine = SyncMLCEngine(
    model=str(MODEL),
    model_lib=str(lib_path),
    device=DEV,
    mode="interactive",
    engine_config=engine_config,
    request_stream_callback=request_stream_callback,
)
request = engine.create_request(
    request_id="pd-engine-external-kv",
    inputs=[ExternalKVData(k_data, v_data, prefill_length, [seed_token])],
    generation_config=GenerationConfig(max_tokens=max_tokens, temperature=0.0),
)
engine.add_request(request)

for step in range(16):
    engine.step()
    if finish_reasons:
        break
else:
    raise RuntimeError("Engine request did not finish within 16 steps")

print(f"engine emitted tokens={emitted_token_ids}")
ok = emitted_token_ids == reference_tokens
print(f"\n=== ENGINE EXTERNAL-KV PD {'PASS' if ok else 'FAIL'} ===")
print(f"expected={reference_tokens} actual={emitted_token_ids} emitted_count={len(emitted_token_ids)}")
sys.exit(0 if ok else 1)

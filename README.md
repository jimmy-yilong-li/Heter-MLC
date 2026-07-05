<p align="center">
  <img src="./logo_HeteroMLC.png" alt="Heter-MLC project logo" width="260">
</p>

# Hetero-MLC

Heter-MLC is an open-source project for heterogeneous prefill/decode (PD) split
serving on top of **MLC-LLM**. It extends the MLC serving path so prefill can be
produced by an external accelerator while decode continues through the normal
MLC engine.

The core idea is simple: an external producer, such as CoreML/ANE on Apple
Silicon, produces prefill K/V tensors; Heter-MLC injects those tensors into
MLC's paged KV cache; MLC then continues decode on Metal through its normal
engine path. The result is a focused MLC extension for cross-accelerator
serving.

## What This Adds

- A paged-KV injection primitive, `DebugSetKV`, that writes externally produced
  K/V tensors into an existing sequence cache.
- An MLC serving data type, `ExternalKVData`, for backend-neutral K/V handoff.
- An MLC engine action, `PDExternalKVPrefill`, that reserves pages, injects K/V,
  seeds decode tokens, and hands control back to normal `BatchDecode`.
- A cleaned-up physical-position-map helper shared by debug KV paths.

## MLC Integration

Heter-MLC references an external MLC-LLM checkout instead of vendoring a full
MLC source tree. The integration code in this repository is laid out as an
overlay for MLC-LLM:

- `src/mlc-llm/cpp/...` contains the serving-engine integration.
- `src/mlc-llm/python/...` contains the Python serving API integration.
- `src/mlc-llm/3rdparty/tvm/...` contains the bundled runtime/cache updates
  used by MLC.
- `examples/qwen3_engine_external_kv_pd.py` is the engine-level smoke driver.

Example layout:

```bash
git clone https://github.com/mlc-ai/mlc-llm.git
git clone https://github.com/YOUR_ORG/Heter-MLC.git

rsync -a Heter-MLC/src/mlc-llm/ mlc-llm/
cp Heter-MLC/examples/qwen3_engine_external_kv_pd.py .
```

The overlay mirrors the target paths in an MLC-LLM checkout. If upstream MLC
moves, rebase the overlay against the new checkout before using it.

## Current Progress

The current implementation has been validated on the local Apple Silicon
development path:

- MLC-LLM build passes.
- `DebugSetKV` tests pass.
- `PDExternalKVPrefill` C++ mock-action tests pass.
- Engine-level external-KV smoke passes on Qwen3-0.6B with exact two-token
  parity against a VM reference continuation.

The first public boundary is intentionally narrow: pure-text LLM serving,
single decode model, paged KV cache, and a Python-side external K/V producer.
CoreML/ANE stays outside the C++ engine as a producer boundary.

## Next Steps

Detailed planning is tracked outside this public package during early
development. For the public repo, the next work should be tracked as GitHub
Issues or a GitHub Project. The immediate items are:

- Build a variable-length CoreML/ANE K/V producer.
- Integrate external-KV requests with MLC's prefix-cache path once the producer
  supplies a stable prefix identity or prompt-token record.
- Run longer-sequence latency sweeps to prioritize zero-copy handoff,
  injection-kernel optimization, and page-size tuning.
- Decide the stable public API name for the K/V injection primitive and add
  deeper validation once sampler-side reporting is suitable.

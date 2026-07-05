/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/pd_external_kv.cc
 */

#include <tvm/support/cuda/nvtx.h>

#include <chrono>
#include <limits>

#include "../data.h"
#include "../metrics.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::support::NVTXScopedRange;

/*!
 * \brief Inject externally produced K/V for a pending request, then seed the existing decode path.
 *
 * This action intentionally keeps producer logic out of C++. Python can run CoreML/ANE or any
 * other producer and submit backend-neutral ExternalKVData. The action only reserves KV pages,
 * calls Model::DebugSetKV, commits seed decode token(s), and lets BatchDecode do logits/sampling.
 */
class PDExternalKVPrefillActionObj : public EngineActionObj {
 public:
  explicit PDExternalKVPrefillActionObj(Array<Model> models, EngineConfig engine_config,
                                        Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {
    TVM_FFI_ICHECK_EQ(models_.size(), 1U)
        << "External K/V injection currently supports a single decode model only.";
    TVM_FFI_ICHECK(models_[0]->GetMetadata().kv_state_kind == KVStateKind::kKVCache)
        << "External K/V injection currently supports PagedKVCache models only.";
  }

  Array<Request> Step(EngineState estate) final {
    if (estate->waiting_queue.empty() ||
        !ContainsExternalKVData(estate->waiting_queue.front()->inputs)) {
      return {};
    }

    auto it = estate->waiting_queue.begin();
    Request request = *it;
    const auto* request_external_kv = GetSoleExternalKVData(request->inputs);
    TVM_FFI_ICHECK(request_external_kv != nullptr)
        << "ExternalKVData requests must contain exactly one input: ExternalKVData.";
    if (!estate->running_queue.empty()) {
      // Keep the first version serialized with running decode requests. The injected prefix and
      // replay tokens are then handed to BatchDecode in a predictable FIFO order.
      return {};
    }

    RequestState rstate = estate->GetRequestState(request);
    RequestStateEntry rsentry = GetSolePendingEntry(rstate);
    RequestModelState mstate = GetSoleModelState(rsentry);
    const auto* external_kv = GetExternalKVForState(mstate, request_external_kv);
    TVM_FFI_ICHECK(estate->prefix_cache->Mode() == PrefixCacheMode::kDisable)
        << "External K/V injection requires prefix_cache_mode=\"disable\" because the engine "
           "does not have prompt token ids to register in the prefix cache.";
    TVM_FFI_ICHECK_EQ(request->generation_cfg->n, 1)
        << "External K/V injection currently supports n=1 only.";

    bool initial_injection = mstate->committed_tokens.empty();
    int prefill_length = external_kv->prefill_length;
    int num_replay_tokens = GetNumReplayTokens(mstate, external_kv, initial_injection);
    ReserveKVCapacity(estate, prefill_length, num_replay_tokens);

    auto tstart = std::chrono::high_resolution_clock::now();
    InjectAndSeedDecode(estate, it, request, rstate, rsentry, mstate, external_kv,
                        initial_injection, num_replay_tokens);
    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    // Return no processed request so BatchDecode runs in the same Engine::Step.
    return {};
  }

 private:
  RequestStateEntry GetSolePendingEntry(const RequestState& rstate) const {
    TVM_FFI_ICHECK_EQ(rstate->entries.size(), 1U)
        << "External K/V injection does not support branched request state entries.";
    RequestStateEntry rsentry = rstate->entries[0];
    TVM_FFI_ICHECK(rsentry->status == RequestStateStatus::kPending)
        << "External K/V injection expects a pending request state entry.";
    TVM_FFI_ICHECK_EQ(rsentry->parent_idx, -1)
        << "External K/V injection does not support prefix-cache fork/resume yet.";
    TVM_FFI_ICHECK(rsentry->child_indices.empty())
        << "External K/V injection does not support parallel child branches yet.";
    return rsentry;
  }

  RequestModelState GetSoleModelState(const RequestStateEntry& rsentry) const {
    TVM_FFI_ICHECK_EQ(rsentry->mstates.size(), 1U);
    RequestModelState mstate = rsentry->mstates[0];
    TVM_FFI_ICHECK(!mstate->inputs.empty());
    return mstate;
  }

  const ExternalKVDataNode* GetExternalKVForState(
      const RequestModelState& mstate, const ExternalKVDataNode* request_external_kv) const {
    const auto* external_kv = mstate->inputs[0].as<ExternalKVDataNode>();
    TVM_FFI_ICHECK_NOTNULL(external_kv);
    for (int i = 1; i < static_cast<int>(mstate->inputs.size()); ++i) {
      TVM_FFI_ICHECK(mstate->inputs[i].as<TokenDataNode>() != nullptr)
          << "External K/V rehydration only supports token replay inputs after ExternalKVData.";
    }
    TVM_FFI_ICHECK(external_kv == request_external_kv)
        << "External K/V request state does not match the request input.";
    return external_kv;
  }

  int GetNumReplayTokens(const RequestModelState& mstate, const ExternalKVDataNode* external_kv,
                         bool initial_injection) const {
    int num_replay_tokens = initial_injection
                                ? static_cast<int>(external_kv->decode_token_ids.size())
                                : static_cast<int>(mstate->committed_tokens.size());
    TVM_FFI_ICHECK_GT(external_kv->prefill_length, 0);
    TVM_FFI_ICHECK_GT(num_replay_tokens, 0);
    if (initial_injection) {
      for (int64_t token_id : external_kv->decode_token_ids) {
        TVM_FFI_ICHECK_GE(token_id, std::numeric_limits<int32_t>::min());
        TVM_FFI_ICHECK_LE(token_id, std::numeric_limits<int32_t>::max());
      }
    }
    return num_replay_tokens;
  }

  int CountPagesForTokens(int num_tokens) const {
    return (num_tokens + engine_config_->kv_cache_page_size - 1) /
           engine_config_->kv_cache_page_size;
  }

  void ReserveKVCapacity(EngineState estate, int prefill_length, int num_replay_tokens) const {
    int required_pages = CountPagesForTokens(prefill_length);
    // BatchDecode will immediately append replay token(s), so keep those pages available.
    required_pages += CountPagesForTokens(num_replay_tokens);
    while (models_[0]->GetNumAvailablePages() < required_pages) {
      TVM_FFI_ICHECK(estate->prefix_cache->TryFreeMemory())
          << "External K/V injection cannot reserve enough KV pages for prefill plus replay.";
    }
    TVM_FFI_ICHECK_LE(
        models_[0]->GetCurrentTotalSequenceLength() + prefill_length + num_replay_tokens,
        engine_config_->max_total_sequence_length)
        << "External K/V injection would exceed max_total_sequence_length.";
  }

  void InjectAndSeedDecode(EngineState estate, std::vector<Request>::iterator waiting_it,
                           Request request, RequestState rstate, RequestStateEntry rsentry,
                           RequestModelState mstate, const ExternalKVDataNode* external_kv,
                           bool initial_injection, int num_replay_tokens) {
    int prefill_length = external_kv->prefill_length;

    NVTXScopedRange nvtx_scope("PDExternalKVPrefill inject request " + request->id);
    RECORD_EVENT(trace_recorder_, request->id, "start external kv inject");

    // After this point the action follows the existing fatal-check prefill contract. The first
    // version does not attempt request-level rollback if AddNewSequence/DebugSetKV preconditions
    // fail after engine state starts mutating.
    rsentry->status = RequestStateStatus::kAlive;
    estate->running_queue.push_back(request);

    models_[0]->AddNewSequence(mstate->internal_id);
    models_[0]->EnableSlidingWindowForSeq(mstate->internal_id);
    models_[0]->DisaggPrepareKVRecv(mstate->internal_id, prefill_length);
    models_[0]->DebugSetKV(mstate->internal_id, 0, external_kv->k_data, external_kv->v_data);

    mstate->inputs.clear();
    mstate->num_prefilled_tokens += prefill_length;
    if (initial_injection) {
      for (int64_t token_id : external_kv->decode_token_ids) {
        mstate->CommitToken({{static_cast<int32_t>(token_id), 1.0f}, {}});
      }
      // Seed tokens are decode inputs, not generated deltas to stream back.
      rsentry->next_callback_token_pos = static_cast<int>(mstate->committed_tokens.size());
      if (request->generation_cfg->max_tokens != -1) {
        ObjectPtr<GenerationConfigNode> updated_generation_cfg =
            tvm::ffi::make_object<GenerationConfigNode>(*request->generation_cfg.get());
        updated_generation_cfg->max_tokens += num_replay_tokens;
        request->generation_cfg = GenerationConfig(updated_generation_cfg);
      }
      rstate->metrics.prefill_tokens += prefill_length;
      rstate->metrics.prefill_end_time_point = std::chrono::high_resolution_clock::now();
    } else {
      mstate->num_tokens_for_next_decode = num_replay_tokens;
    }
    mstate->cached_committed_tokens = mstate->committed_tokens.size();

    estate->waiting_queue.erase(waiting_it);
    estate->running_rsentries_changed = true;
    RECORD_EVENT(trace_recorder_, request->id, "finish external kv inject");
  }

  Array<Model> models_;
  EngineConfig engine_config_;
  Optional<EventTraceRecorder> trace_recorder_;
};

EngineAction EngineAction::PDExternalKVPrefill(Array<Model> models, EngineConfig engine_config,
                                               Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<PDExternalKVPrefillActionObj>(
      std::move(models), std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

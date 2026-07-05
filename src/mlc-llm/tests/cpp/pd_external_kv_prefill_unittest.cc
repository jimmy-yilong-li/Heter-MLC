#include <gtest/gtest.h>

#include "serve/engine_actions/action.h"
#include "serve/prefix_cache.h"

namespace mlc {
namespace llm {
namespace serve {
namespace {

using tvm::ffi::make_object;
using tvm::ffi::Shape;
using tvm::runtime::Tensor;

class FakeModelObj : public ModelObj {
 public:
  ObjectRef TokenEmbed(Shape, ObjectRef*, int) final { LOG(FATAL) << "unused"; }
  ObjectRef ImageEmbed(const Tensor&, ObjectRef*, int) final { LOG(FATAL) << "unused"; }
  ObjectRef FuseEmbedHidden(const ObjectRef&, const ObjectRef&, int, int) final {
    LOG(FATAL) << "unused";
  }
  bool CanGetLogits() final { return true; }
  Tensor GetLogits(const ObjectRef&) final { LOG(FATAL) << "unused"; }
  Array<Tensor> GetMultiStepLogits(const ObjectRef&) final { LOG(FATAL) << "unused"; }
  Tensor BatchPrefill(const ObjectRef&, const std::vector<int64_t>&,
                      const std::vector<int>&) final {
    LOG(FATAL) << "unused";
  }
  ObjectRef BatchPrefillToLastHidden(const ObjectRef&, const std::vector<int64_t>&,
                                     const std::vector<int>&) final {
    LOG(FATAL) << "unused";
  }
  Tensor BatchDecode(const ObjectRef&, const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  Tensor BatchTreeDecode(const ObjectRef&, const std::vector<int64_t>&, const std::vector<int>&,
                         const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  ObjectRef BatchDecodeToLastHidden(const ObjectRef&, const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  Tensor BatchVerify(const ObjectRef&, const std::vector<int64_t>&, const std::vector<int>&,
                     const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  ObjectRef BatchVerifyToLastHidden(const ObjectRef&, const std::vector<int64_t>&,
                                    const std::vector<int>&, const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  void CreateKVCache(int, int, int64_t, int64_t, int, int) final {}
  void AddNewSequence(int64_t seq_id) final {
    add_new_sequence_calls++;
    added_seq_id = seq_id;
  }
  void ForkSequence(int64_t, int64_t, int64_t) final { LOG(FATAL) << "unused"; }
  void RemoveSequence(int64_t) final { LOG(FATAL) << "unused"; }
  void PopNFromKVCache(int64_t, int) final { LOG(FATAL) << "unused"; }
  void CommitAcceptedTokenTreeNodesToKVCache(const std::vector<int64_t>&,
                                             const std::vector<int64_t>&) final {
    LOG(FATAL) << "unused";
  }
  void EnableSlidingWindowForSeq(int64_t seq_id) final {
    enable_sliding_window_calls++;
    sliding_window_seq_id = seq_id;
  }
  Shape DisaggPrepareKVRecv(int64_t seq_id, int length) final {
    disagg_prepare_calls++;
    disagg_seq_id = seq_id;
    disagg_length = length;
    return Shape{};
  }
  void DisaggMarkKVSend(int64_t, int, Shape, int) final { LOG(FATAL) << "unused"; }
  void DebugSetKV(int64_t seq_id, int64_t start_pos, const Tensor& k_data,
                  const Tensor& v_data) final {
    debug_set_kv_calls++;
    debug_seq_id = seq_id;
    debug_start_pos = start_pos;
    debug_k_data = k_data;
    debug_v_data = v_data;
  }
  ModelMetadata GetMetadata() const final {
    ModelMetadata metadata;
    metadata.kv_state_kind = KVStateKind::kKVCache;
    metadata.kv_cache_metadata.num_hidden_layers = 2;
    metadata.kv_cache_metadata.num_key_value_heads = 1;
    metadata.kv_cache_metadata.head_dim = 4;
    return metadata;
  }
  int GetNumAvailablePages() const final { return num_available_pages; }
  int GetCurrentTotalSequenceLength() const final { return current_total_sequence_length; }
  void LoadParams() final {}
  void SetMaxNumSequence(int) final {}
  void SetPrefillChunkSize(int) final {}
  LogitProcessor CreateLogitProcessor(int, Optional<EventTraceRecorder>) final {
    LOG(FATAL) << "unused";
  }
  Sampler CreateSampler(int, int, Optional<EventTraceRecorder>) final { LOG(FATAL) << "unused"; }
  int EstimateHostCPURequirement() const final { return 0; }
  int GetSlidingWindowSize() const final { return -1; }
  int GetAttentionSinkSize() const final { return 0; }
  ObjectRef AllocEmbeddingTensor() final { LOG(FATAL) << "unused"; }
  ObjectRef AllocHiddenStatesTensor() final { LOG(FATAL) << "unused"; }
  void Reset() final {}
  DraftTokenWorkspaceManager CreateDraftTokenWorkspaceManager(int) final { LOG(FATAL) << "unused"; }
  ObjectRef GatherHiddenStates(const ObjectRef&, const std::vector<int>&, ObjectRef*) final {
    LOG(FATAL) << "unused";
  }
  void ScatterHiddenStates(const ObjectRef&, const std::vector<int>&, ObjectRef*) final {
    LOG(FATAL) << "unused";
  }
  Tensor GatherDraftProbs(const Tensor&, const std::vector<int>&, Tensor*) final {
    LOG(FATAL) << "unused";
  }
  void ScatterDraftProbs(const Tensor&, const std::vector<int>&, Tensor*) final {
    LOG(FATAL) << "unused";
  }
  void DebugCallFuncOnAllAllWorker(const String&, Optional<String>) final {}

  int num_available_pages = 8;
  int current_total_sequence_length = 0;
  int add_new_sequence_calls = 0;
  int enable_sliding_window_calls = 0;
  int disagg_prepare_calls = 0;
  int debug_set_kv_calls = 0;
  int64_t added_seq_id = -1;
  int64_t sliding_window_seq_id = -1;
  int64_t disagg_seq_id = -1;
  int disagg_length = -1;
  int64_t debug_seq_id = -1;
  int64_t debug_start_pos = -1;
  Tensor debug_k_data{nullptr};
  Tensor debug_v_data{nullptr};
};

Model MakeModel(ObjectPtr<FakeModelObj>* model_obj) {
  *model_obj = make_object<FakeModelObj>();
  return Model(*model_obj);
}

EngineConfig MakeConfig(PrefixCacheMode prefix_cache_mode = PrefixCacheMode::kDisable) {
  ObjectPtr<EngineConfigNode> n = make_object<EngineConfigNode>();
  n->kv_cache_page_size = 4;
  n->max_total_sequence_length = 64;
  n->max_single_sequence_length = 64;
  n->prefill_chunk_size = 64;
  n->prefix_cache_mode = prefix_cache_mode;
  return EngineConfig(n);
}

GenerationConfig MakeGenerationConfig(int max_tokens = 2) {
  ObjectPtr<GenerationConfigNode> n = make_object<GenerationConfigNode>();
  n->n = 1;
  n->max_tokens = max_tokens;
  n->debug_config.disagg_config.kind = DisaggRequestKind::kNone;
  return GenerationConfig(n);
}

ExternalKVData MakeExternalKVData(int prefill_length = 5, Shape decode_token_ids = Shape{42}) {
  DLDevice device{kDLCPU, 0};
  Tensor k_data = Tensor::Empty({2, prefill_length, 1, 4}, DLDataType{kDLFloat, 32, 1}, device);
  Tensor v_data = Tensor::Empty({2, prefill_length, 1, 4}, DLDataType{kDLFloat, 32, 1}, device);
  return ExternalKVData(k_data, v_data, prefill_length, decode_token_ids);
}

RequestState AttachRequestState(EngineState estate, const Request& request, int64_t internal_id) {
  RequestStateEntry entry(request, 1, internal_id, /*rng_seed=*/0, /*token_table=*/{},
                          /*compiled_grammar=*/std::nullopt);
  RequestState rstate({entry}, request->generation_cfg->n,
                      std::chrono::high_resolution_clock::now());
  request->rstate = rstate.get();
  estate->request_states[request->id] = rstate;
  return rstate;
}

Request MakeRequest(String id, ExternalKVData external_kv, int max_tokens = 2) {
  return Request(id, Array<Data>{external_kv}, MakeGenerationConfig(max_tokens));
}

TEST(PDExternalKVPrefillActionTest, InjectsExternalKVAndSeedsDecodeTokens) {
  ObjectPtr<FakeModelObj> model_obj;
  Model model = MakeModel(&model_obj);
  EngineConfig config = MakeConfig();
  EngineState estate;
  estate->prefix_cache = PrefixCache::CreateNoPrefixCache();
  ExternalKVData external_kv = MakeExternalKVData(/*prefill_length=*/5, Shape{42, 43});
  Request request = MakeRequest("req", external_kv, /*max_tokens=*/3);
  RequestState rstate = AttachRequestState(estate, request, /*internal_id=*/7);
  estate->waiting_queue.push_back(request);

  EngineAction action =
      EngineAction::PDExternalKVPrefill({model}, config, Optional<EventTraceRecorder>());
  Array<Request> processed = action->Step(estate);

  EXPECT_EQ(processed.size(), 0U);
  EXPECT_TRUE(estate->waiting_queue.empty());
  ASSERT_EQ(estate->running_queue.size(), 1U);
  EXPECT_TRUE(estate->running_queue[0].same_as(request));
  EXPECT_TRUE(estate->running_rsentries_changed);
  RequestStateEntry entry = rstate->entries[0];
  RequestModelState mstate = entry->mstates[0];
  EXPECT_EQ(entry->status, RequestStateStatus::kAlive);
  EXPECT_EQ(mstate->num_prefilled_tokens, 5);
  ASSERT_EQ(mstate->committed_tokens.size(), 2U);
  EXPECT_EQ(mstate->committed_tokens[0].GetTokenId(), 42);
  EXPECT_EQ(mstate->committed_tokens[1].GetTokenId(), 43);
  EXPECT_EQ(entry->next_callback_token_pos, 2);
  EXPECT_EQ(request->generation_cfg->max_tokens, 5);
  EXPECT_TRUE(mstate->inputs.empty());
  EXPECT_EQ(mstate->cached_committed_tokens, 2);
  EXPECT_EQ(model_obj->add_new_sequence_calls, 1);
  EXPECT_EQ(model_obj->enable_sliding_window_calls, 1);
  EXPECT_EQ(model_obj->disagg_prepare_calls, 1);
  EXPECT_EQ(model_obj->debug_set_kv_calls, 1);
  EXPECT_EQ(model_obj->added_seq_id, 7);
  EXPECT_EQ(model_obj->sliding_window_seq_id, 7);
  EXPECT_EQ(model_obj->disagg_seq_id, 7);
  EXPECT_EQ(model_obj->disagg_length, 5);
  EXPECT_EQ(model_obj->debug_seq_id, 7);
  EXPECT_EQ(model_obj->debug_start_pos, 0);
  EXPECT_TRUE(model_obj->debug_k_data.same_as(external_kv->k_data));
  EXPECT_TRUE(model_obj->debug_v_data.same_as(external_kv->v_data));
}

TEST(PDExternalKVPrefillActionTest, DoesNothingWhenExternalKVRequestIsNotAtQueueFront) {
  ObjectPtr<FakeModelObj> model_obj;
  Model model = MakeModel(&model_obj);
  EngineConfig config = MakeConfig();
  EngineState estate;
  estate->prefix_cache = PrefixCache::CreateNoPrefixCache();
  Request text_request =
      Request("text", Array<Data>{TokenData(std::vector<int32_t>{1, 2})}, MakeGenerationConfig());
  Request external_request = MakeRequest("external", MakeExternalKVData(), /*max_tokens=*/2);
  AttachRequestState(estate, text_request, /*internal_id=*/1);
  AttachRequestState(estate, external_request, /*internal_id=*/2);
  estate->waiting_queue.push_back(text_request);
  estate->waiting_queue.push_back(external_request);

  EngineAction action =
      EngineAction::PDExternalKVPrefill({model}, config, Optional<EventTraceRecorder>());
  Array<Request> processed = action->Step(estate);

  EXPECT_EQ(processed.size(), 0U);
  ASSERT_EQ(estate->waiting_queue.size(), 2U);
  EXPECT_TRUE(estate->waiting_queue[0].same_as(text_request));
  EXPECT_TRUE(estate->waiting_queue[1].same_as(external_request));
  EXPECT_TRUE(estate->running_queue.empty());
  EXPECT_EQ(model_obj->debug_set_kv_calls, 0);
}

TEST(PDExternalKVPrefillActionTest, WaitsForRunningDecodeRequests) {
  ObjectPtr<FakeModelObj> model_obj;
  Model model = MakeModel(&model_obj);
  EngineConfig config = MakeConfig();
  EngineState estate;
  estate->prefix_cache = PrefixCache::CreateNoPrefixCache();
  Request running_request =
      Request("running", Array<Data>{TokenData(std::vector<int32_t>{1})}, MakeGenerationConfig());
  Request external_request = MakeRequest("external", MakeExternalKVData(), /*max_tokens=*/2);
  AttachRequestState(estate, running_request, /*internal_id=*/1);
  AttachRequestState(estate, external_request, /*internal_id=*/2);
  estate->running_queue.push_back(running_request);
  estate->waiting_queue.push_back(external_request);

  EngineAction action =
      EngineAction::PDExternalKVPrefill({model}, config, Optional<EventTraceRecorder>());
  Array<Request> processed = action->Step(estate);

  EXPECT_EQ(processed.size(), 0U);
  ASSERT_EQ(estate->waiting_queue.size(), 1U);
  EXPECT_TRUE(estate->waiting_queue[0].same_as(external_request));
  ASSERT_EQ(estate->running_queue.size(), 1U);
  EXPECT_TRUE(estate->running_queue[0].same_as(running_request));
  EXPECT_EQ(model_obj->debug_set_kv_calls, 0);
}

TEST(PDExternalKVPrefillActionTest, RejectsRadixPrefixCacheMode) {
  ObjectPtr<FakeModelObj> model_obj;
  Model model = MakeModel(&model_obj);
  EngineConfig config = MakeConfig(PrefixCacheMode::kRadix);
  EngineState estate;
  estate->prefix_cache = PrefixCache::CreateRadixPrefixCache(/*max_recycling_seqs=*/0);
  Request request = MakeRequest("external", MakeExternalKVData(), /*max_tokens=*/2);
  AttachRequestState(estate, request, /*internal_id=*/3);
  estate->waiting_queue.push_back(request);

  EngineAction action =
      EngineAction::PDExternalKVPrefill({model}, config, Optional<EventTraceRecorder>());
  EXPECT_THROW({ action->Step(estate); }, tvm::ffi::Error);
  EXPECT_EQ(model_obj->debug_set_kv_calls, 0);
}

}  // namespace
}  // namespace serve
}  // namespace llm
}  // namespace mlc

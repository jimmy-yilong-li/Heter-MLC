/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/data.cc
 */
#include "data.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <limits>

#include "model.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_FFI_STATIC_INIT_BLOCK() {
  DataNode::RegisterReflection();
  TextDataNode::RegisterReflection();
  TokenDataNode::RegisterReflection();
  ImageDataNode::RegisterReflection();
  ExternalKVDataNode::RegisterReflection();
  RequestStreamOutputObj::RegisterReflection();
}

/****************** Data ******************/

std::pair<Array<Data>, Array<Data>> SplitData(const Array<Data>& original_data, int total_length,
                                              int split_pos) {
  TVM_FFI_ICHECK_GE(split_pos, 0);
  TVM_FFI_ICHECK_GE(total_length, split_pos)
      << "Cannot truncate when the current length is already less than the target length";
  std::vector<Data> lhs(original_data.begin(), original_data.end());
  std::vector<Data> rhs;
  while (total_length > split_pos) {
    TVM_FFI_ICHECK(!lhs.empty());
    Data last_data = lhs.back();
    int last_data_length = last_data->GetLength();
    TVM_FFI_ICHECK_GE(total_length - last_data_length, 0);
    if (total_length - last_data_length >= split_pos) {
      // Pop the entire last data.
      rhs.push_back(lhs.back());
      lhs.pop_back();
      total_length -= last_data_length;
      continue;
    }
    // Partially truncate the last data.
    const auto* token_data = last_data.as<TokenDataNode>();
    TVM_FFI_ICHECK(token_data != nullptr) << "Only TokenData supports partial truncation.";
    int length_to_truncate = total_length - split_pos;
    TVM_FFI_ICHECK_GT(length_to_truncate, 0);
    TVM_FFI_ICHECK_LT(length_to_truncate, last_data_length);
    TokenData lhs_token_data(
        Shape{token_data->token_ids.begin(), token_data->token_ids.end() - length_to_truncate});
    TokenData rhs_token_data(
        Shape{token_data->token_ids.end() - length_to_truncate, token_data->token_ids.end()});
    TVM_FFI_ICHECK_EQ(total_length - last_data_length + lhs_token_data->GetLength(), split_pos);
    lhs.pop_back();
    lhs.push_back(lhs_token_data);
    rhs.push_back(rhs_token_data);
    std::reverse(rhs.begin(), rhs.end());
    total_length = split_pos;
  }
  return {lhs, rhs};
}

/****************** TextData ******************/

TextData::TextData(String text) {
  ObjectPtr<TextDataNode> n = tvm::ffi::make_object<TextDataNode>();
  n->text = std::move(text);
  data_ = std::move(n);
}

int TextDataNode::GetLength() const {
  LOG(FATAL) << "\"GetLength\" for TextData is not supported. "
                "Please tokenize the text and construct a TokenData object.";
}

ObjectRef TextDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  LOG(FATAL) << "\"GetEmbedding\" for TextData is not supported. "
                "Please tokenize the text and construct a TokenData object.";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("mlc.serve.TextData", [](String text) { return TextData(std::move(text)); })
      .def("mlc.serve.TextDataGetTextString", [](TextData data) { return data->text; });
}

/****************** TokenData ******************/

TokenData::TokenData(Shape token_ids) {
  ObjectPtr<TokenDataNode> n = tvm::ffi::make_object<TokenDataNode>();
  n->token_ids = std::move(token_ids);
  data_ = std::move(n);
}

TokenData::TokenData(std::vector<int32_t> token_ids) {
  ObjectPtr<TokenDataNode> n = tvm::ffi::make_object<TokenDataNode>();
  n->token_ids = Shape(token_ids.begin(), token_ids.end());
  data_ = std::move(n);
}

int TokenDataNode::GetLength() const { return token_ids.size(); }

ObjectRef TokenDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  return model->TokenEmbed(token_ids, dst, offset);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("mlc.serve.TokenData",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    std::vector<int32_t> token_ids;
                    token_ids.reserve(args.size());
                    for (int i = 0; i < args.size(); i++) {
                      token_ids.push_back(args[i].cast<int32_t>());
                    }
                    *rv = TokenData(std::move(token_ids));
                  })
      .def("mlc.serve.TokenDataGetTokenIds", [](TokenData data) { return data->token_ids; });
}

/****************** ImageData ******************/

ImageData::ImageData(Tensor image, int embed_size) {
  ObjectPtr<ImageDataNode> n = tvm::ffi::make_object<ImageDataNode>();
  n->image = std::move(image);
  n->embed_size = embed_size;
  data_ = std::move(n);
}

int ImageDataNode::GetLength() const { return embed_size; }

ObjectRef ImageDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  return model->ImageEmbed(image, dst, offset);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("mlc.serve.ImageData",
           [](Tensor image, int embed_size) { return ImageData(std::move(image), embed_size); })
      .def("mlc.serve.ImageDataGetImage", [](ImageData data) { return data->image; });
}

/****************** ExternalKVData ******************/

ExternalKVData::ExternalKVData(Tensor k_data, Tensor v_data, int prefill_length,
                               Shape decode_token_ids) {
  TVM_FFI_ICHECK(k_data.defined()) << "ExternalKVData expects defined K tensor.";
  TVM_FFI_ICHECK(v_data.defined()) << "ExternalKVData expects defined V tensor.";
  TVM_FFI_ICHECK_EQ(k_data->ndim, 4)
      << "ExternalKVData expects K layout (num_layers, prefill_length, num_kv_heads, head_dim).";
  TVM_FFI_ICHECK_EQ(v_data->ndim, 4)
      << "ExternalKVData expects V layout (num_layers, prefill_length, num_kv_heads, head_dim).";
  TVM_FFI_ICHECK_GT(prefill_length, 0);
  TVM_FFI_ICHECK_EQ(k_data->shape[1], prefill_length);
  TVM_FFI_ICHECK_EQ(v_data->shape[1], prefill_length);
  TVM_FFI_ICHECK_EQ(k_data->shape[0], v_data->shape[0])
      << "ExternalKVData expects K and V to have the same number of layers.";
  TVM_FFI_ICHECK_EQ(k_data->shape[2], v_data->shape[2])
      << "ExternalKVData expects K and V to have the same number of KV heads.";
  TVM_FFI_ICHECK_EQ(k_data->shape[3], v_data->shape[3])
      << "ExternalKVData expects K and V to have the same head dimension.";
  TVM_FFI_ICHECK(k_data.DataType() == v_data.DataType())
      << "ExternalKVData expects K and V to have the same dtype.";
  TVM_FFI_ICHECK_GT(decode_token_ids.size(), 0)
      << "ExternalKVData expects at least one decode token id.";
  for (int64_t token_id : decode_token_ids) {
    TVM_FFI_ICHECK_GE(token_id, std::numeric_limits<int32_t>::min());
    TVM_FFI_ICHECK_LE(token_id, std::numeric_limits<int32_t>::max());
  }

  ObjectPtr<ExternalKVDataNode> n = tvm::ffi::make_object<ExternalKVDataNode>();
  n->k_data = std::move(k_data);
  n->v_data = std::move(v_data);
  n->prefill_length = prefill_length;
  n->decode_token_ids = std::move(decode_token_ids);
  data_ = std::move(n);
}

int ExternalKVDataNode::GetLength() const { return prefill_length; }

ObjectRef ExternalKVDataNode::GetEmbedding(Model model, ObjectRef* dst, int offset) const {
  LOG(FATAL) << "\"GetEmbedding\" for ExternalKVData is not supported. "
                "External K/V must be consumed by the PD external-KV engine action.";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("mlc.serve.ExternalKVData",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK_GE(args.size(), 4)
                        << "ExternalKVData expects k, v, prefill_length, and decode tokens.";
                    std::vector<int64_t> decode_token_ids;
                    decode_token_ids.reserve(args.size() - 3);
                    for (int i = 3; i < args.size(); ++i) {
                      decode_token_ids.push_back(args[i].cast<int64_t>());
                    }
                    *rv = ExternalKVData(args[0].cast<Tensor>(), args[1].cast<Tensor>(),
                                         args[2].cast<int>(),
                                         Shape(decode_token_ids.begin(), decode_token_ids.end()));
                  })
      .def("mlc.serve.ExternalKVDataGetK", [](ExternalKVData data) { return data->k_data; })
      .def("mlc.serve.ExternalKVDataGetV", [](ExternalKVData data) { return data->v_data; })
      .def("mlc.serve.ExternalKVDataGetDecodeTokenIds",
           [](ExternalKVData data) { return data->decode_token_ids; });
}

/****************** SampleResult ******************/

/*! \brief Convert a single token with probability to JSON string. */
inline void TokenToLogProbJSON(const Tokenizer& tokenizer, const TokenProbPair& token_prob,
                               std::ostringstream* os) {
  const std::string& token = tokenizer->PostProcessedTokenTable()[token_prob.first];

  (*os) << "\"token\": \"";
  for (char ch : token) {
    if (ch >= 33 && ch <= 126) {
      // The character is in ASCII visible range.
      // Handle escape characters in JSON.
      if (ch == '"') {
        (*os) << "\\\"";
      } else if (ch == '\\') {
        (*os) << "\\\\";
      } else {
        (*os) << ch;
      }
    }
  }
  (*os) << "\", ";
  (*os) << "\"logprob\": " << std::log(std::max(token_prob.second, 1e-10f)) << ", ";
  (*os) << "\"bytes\": [";
  int token_len = token.size();
  for (int pos = 0; pos < token_len; ++pos) {
    (*os) << static_cast<int>(static_cast<unsigned char>(token[pos]));
    if (pos != token_len - 1) {
      (*os) << ", ";
    }
  }
  (*os) << "]";
}

int32_t SampleResult::GetTokenId() const { return this->sampled_token_id.first; }

std::string SampleResult::GetLogProbJSON(const Tokenizer& tokenizer, bool logprob) const {
  TVM_FFI_ICHECK(top_prob_tokens.empty() || logprob);
  if (!logprob) {
    // Logprob is not needed.
    return "";
  }

  std::ostringstream os;
  os << "{";
  // - Convert the sampled token to JSON.
  TokenToLogProbJSON(tokenizer, sampled_token_id, &os);
  // - Convert the tokens with top probabilities.
  os << ", \"top_logprobs\": [";
  int num_top = top_prob_tokens.size();
  for (int i = 0; i < num_top; ++i) {
    os << "{";
    TokenToLogProbJSON(tokenizer, top_prob_tokens[i], &os);
    os << "}";
    if (i != num_top - 1) {
      os << ", ";
    }
  }
  os << "]}";
  return os.str();
}

/****************** RequestStreamOutput ******************/

RequestStreamOutput::RequestStreamOutput(
    String request_id, std::vector<std::vector<int64_t>> group_delta_token_ids,
    std::optional<std::vector<std::vector<String>>> group_delta_logprob_json_strs,
    std::vector<Optional<String>> group_finish_reason,
    std::vector<String> group_extra_prefix_string) {
  ObjectPtr<RequestStreamOutputObj> n = tvm::ffi::make_object<RequestStreamOutputObj>();
  n->request_id = std::move(request_id);
  n->group_delta_token_ids = std::move(group_delta_token_ids);
  n->group_delta_logprob_json_strs = std::move(group_delta_logprob_json_strs);
  n->group_finish_reason = std::move(group_finish_reason);
  n->group_extra_prefix_string = std::move(group_extra_prefix_string);
  data_ = std::move(n);
}

RequestStreamOutput RequestStreamOutput::Usage(String request_id,
                                               String request_final_usage_json_str) {
  ObjectPtr<RequestStreamOutputObj> n = tvm::ffi::make_object<RequestStreamOutputObj>();
  n->request_id = std::move(request_id);
  n->request_final_usage_json_str = std::move(request_final_usage_json_str);
  return RequestStreamOutput(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("mlc.serve.RequestStreamOutputUnpack", [](RequestStreamOutput output) {
    TVM_FFI_ICHECK(!output->unpacked)
        << "One RequestStreamOutput can be unpacked for at most once.";
    std::vector<Shape> group_delta_token_ids;
    std::vector<Array<String>> group_delta_logprob_json_strs;
    group_delta_token_ids.reserve(output->group_delta_token_ids.size());
    if (output->group_delta_logprob_json_strs.has_value()) {
      group_delta_logprob_json_strs.reserve(output->group_delta_token_ids.size());
    }
    for (int i = 0; i < static_cast<int>(output->group_delta_token_ids.size()); ++i) {
      group_delta_token_ids.push_back(output->group_delta_token_ids[i]);
      if (output->group_delta_logprob_json_strs.has_value()) {
        group_delta_logprob_json_strs.push_back(output->group_delta_logprob_json_strs.value()[i]);
      }
    }
    Array<Any> ret = {output->request_id,
                      Array<Shape>(std::move(group_delta_token_ids)),
                      output->group_delta_logprob_json_strs.has_value()
                          ? Array<Array<String>>(std::move(group_delta_logprob_json_strs))
                          : Optional<Array<Array<String>>>(),
                      Array<Optional<String>>(output->group_finish_reason),
                      output->request_final_usage_json_str,
                      Array<String>(output->group_extra_prefix_string)};
    output->unpacked = true;
    return ret;
  });
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc

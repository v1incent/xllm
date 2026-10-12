/* Copyright 2025-2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://github.com/jd-opensource/xllm/blob/main/LICENSE
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "c_api/internal/helper.h"
#include "core/framework/config/rec_config.h"
#include "rec.pb.h"

namespace xllm::helper {
namespace {
constexpr const char* kOneRecSparseEmbeddingName = "sparse_embedding";
constexpr const char* kOneRecDecoderContextEmbeddingName =
    "decoder_context_embedding";

int64_t infer_input_tensor_desc_numel(const int64_t* shape, size_t rank) {
  if (shape == nullptr || rank == 0) {
    return -1;
  }

  int64_t numel = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0) {
      return -1;
    }
    if (numel > std::numeric_limits<int64_t>::max() / shape[i]) {
      return -1;
    }
    numel *= shape[i];
  }
  return numel;
}

struct RecEmitRecord {
  int32_t output_index = 0;
  int64_t item_id = 0;
  std::optional<RecItemInfo> item_info;
};

std::vector<RecEmitRecord> collect_rec_emit_records(
    const RequestOutput& req_output) {
  std::vector<RecEmitRecord> emitted_items;
  emitted_items.reserve(req_output.outputs.size());
  const int32_t total_threshold =
      RecConfig::get_instance().total_conversion_threshold();
  for (int32_t i = 0; i < static_cast<int32_t>(req_output.outputs.size());
       ++i) {
    const auto& output = req_output.outputs[i];
    if (!output.item_ids_list.empty()) {
      const bool has_item_infos =
          output.item_infos_list.size() == output.item_ids_list.size();
      for (size_t item_idx = 0; item_idx < output.item_ids_list.size();
           ++item_idx) {
        if (static_cast<int32_t>(emitted_items.size()) >= total_threshold) {
          break;
        }
        std::optional<RecItemInfo> item_info;
        if (has_item_infos) {
          item_info = output.item_infos_list[item_idx];
        }
        RecEmitRecord emitted_item;
        emitted_item.output_index = i;
        emitted_item.item_id = output.item_ids_list[item_idx];
        emitted_item.item_info = std::move(item_info);
        emitted_items.emplace_back(std::move(emitted_item));
      }
    } else if (output.item_ids.has_value() &&
               static_cast<int32_t>(emitted_items.size()) < total_threshold) {
      RecEmitRecord emitted_item;
      emitted_item.output_index = i;
      emitted_item.item_id = output.item_ids.value();
      emitted_item.item_info = output.item_info;
      emitted_items.emplace_back(std::move(emitted_item));
    }
    if (static_cast<int32_t>(emitted_items.size()) >= total_threshold) {
      break;
    }
  }
  return emitted_items;
}

char* duplicate_c_string(const std::string& value) {
  char* out = new char[value.size() + 1];
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

template <typename Value>
XLLM_InferOutputTensor make_numeric_output_tensor(
    const char* name,
    proto::DataType dtype,
    const std::vector<int64_t>& dims,
    const std::vector<Value>& values) {
  auto name_buffer = std::unique_ptr<char[]>(duplicate_c_string(name));
  auto shape = std::make_unique<int64_t[]>(dims.size());
  std::copy(dims.begin(), dims.end(), shape.get());
  auto data = std::make_unique<char[]>(values.size() * sizeof(Value));
  if (!values.empty()) {
    std::memcpy(data.get(), values.data(), values.size() * sizeof(Value));
  }
  XLLM_InferOutputTensor tensor{};
  tensor.name = name_buffer.release();
  tensor.data_type = static_cast<int32_t>(dtype);
  tensor.shape = shape.release();
  tensor.shape_len = dims.size();
  tensor.num_elements = values.size();
  tensor.data = data.release();
  return tensor;
}

XLLM_InferOutputTensor make_string_output_tensor(
    const char* name,
    const std::vector<std::string>& values) {
  size_t packed_size = 0;
  for (const std::string& value : values) {
    if (value.size() > std::numeric_limits<uint32_t>::max() ||
        packed_size > std::numeric_limits<size_t>::max() - sizeof(uint32_t) -
                          value.size()) {
      throw std::length_error("REC string tensor exceeds supported size");
    }
    packed_size += sizeof(uint32_t) + value.size();
  }
  auto name_buffer = std::unique_ptr<char[]>(duplicate_c_string(name));
  auto shape = std::make_unique<int64_t[]>(1);
  shape[0] = static_cast<int64_t>(values.size());
  auto data = std::make_unique<char[]>(packed_size);
  size_t offset = 0;
  for (const std::string& value : values) {
    const uint32_t length = static_cast<uint32_t>(value.size());
    std::memcpy(data.get() + offset, &length, sizeof(length));
    offset += sizeof(length);
    std::memcpy(data.get() + offset, value.data(), value.size());
    offset += value.size();
  }
  XLLM_InferOutputTensor tensor{};
  tensor.name = name_buffer.release();
  tensor.data_type = static_cast<int32_t>(proto::DataType::STRING);
  tensor.shape = shape.release();
  tensor.shape_len = 1;
  tensor.num_elements = values.size();
  tensor.data = data.release();
  return tensor;
}

XLLM_InferOutputTensor make_float_output_tensor(
    const char* name,
    const std::vector<int64_t>& dims,
    const std::vector<float>& values) {
  return make_numeric_output_tensor(name, proto::DataType::FLOAT, dims, values);
}

// OneRec single-step mode: build a "vocab_probs" FLOAT output tensor from the
// per-sequence probability distributions carried on SequenceOutput.embedding
// (routed there by Batch::process_sample_output when max_decode_rounds == 1).
// Each output row is a [vocab] distribution; the emitted tensor is
// [num_outputs, vocab]. Returns false (no entry) when nothing is available.
bool build_vocab_probs_output_tensor(const RequestOutput& req_output,
                                     XLLM_InferOutputTensor* out_entry) {
  if (out_entry == nullptr || req_output.outputs.empty()) {
    return false;
  }
  const size_t num_outputs = req_output.outputs.size();
  int64_t vocab = 0;
  for (const auto& seq_output : req_output.outputs) {
    if (seq_output.embedding.defined() && seq_output.embedding.numel() > 0) {
      vocab = seq_output.embedding.numel();
      break;
    }
  }
  if (vocab <= 0) {
    return false;
  }

  for (const auto& seq_output : req_output.outputs) {
    if (!seq_output.embedding.defined() ||
        seq_output.embedding.numel() != vocab) {
      throw std::invalid_argument(
          "Inconsistent REC vocabulary probability widths");
    }
  }
  std::vector<float> values;
  values.reserve(num_outputs * static_cast<size_t>(vocab));
  for (const auto& seq_output : req_output.outputs) {
    const auto row = seq_output.embedding.to(torch::kCPU, torch::kFloat32)
                         .contiguous()
                         .view({-1});
    const float* data = row.data_ptr<float>();
    values.insert(values.end(), data, data + vocab);
  }

  *out_entry = make_float_output_tensor(
      "vocab_probs", {static_cast<int64_t>(num_outputs), vocab}, values);
  return true;
}

void append_rec_logprobs_to_vector(std::vector<float>* out,
                                   const SequenceOutput& output,
                                   int32_t expected_count) {
  if (out == nullptr || expected_count <= 0) {
    return;
  }
  const auto& token_logprobs = output.token_ids_logprobs;
  const int32_t actual_count = static_cast<int32_t>(token_logprobs.size());
  for (int32_t i = 0; i < expected_count; ++i) {
    if (i < actual_count && token_logprobs[i].has_value()) {
      out->push_back(token_logprobs[i].value());
    } else {
      out->push_back(0.0f);
    }
  }
}

}  // namespace

bool populate_raw_output_tensors(InferenceType inference_type,
                                 const RequestOutput& output,
                                 XLLM_Response* response) {
  if (response == nullptr || inference_type != InferenceType::REC_COMPLETIONS ||
      output.outputs.empty()) {
    return false;
  }
  const auto& config = RecConfig::get_instance();
  const bool convert_items = config.enable_convert_tokens_to_item();
  const size_t rows = output.outputs.size();
  // Attach zero-initialized slots before populating them. The response owns
  // every completed tensor even if a subsequent allocation fails.
  response->output_tensors.entries = new XLLM_InferOutputTensor[5]();
  response->output_tensors.entries_size = 5;
  auto* entries = response->output_tensors.entries;
  size_t count = 0;
  std::vector<RecEmitRecord> items;
  if (convert_items) {
    items = collect_rec_emit_records(output);
    std::vector<int64_t> ids;
    ids.reserve(items.size());
    for (const auto& item : items) {
      ids.emplace_back(item.item_id);
    }
    entries[count++] =
        make_numeric_output_tensor("rec_result",
                                   proto::DataType::INT64,
                                   {static_cast<int64_t>(ids.size())},
                                   ids);
  } else {
    const size_t width = output.outputs.front().token_ids.size();
    std::vector<int32_t> ids;
    ids.reserve(rows * width);
    for (const auto& sequence : output.outputs) {
      if (sequence.token_ids.size() != width) {
        throw std::invalid_argument(
            "REC output token rows have different lengths");
      }
      ids.insert(
          ids.end(), sequence.token_ids.begin(), sequence.token_ids.end());
    }
    entries[count++] = make_numeric_output_tensor(
        "rec_result",
        proto::DataType::INT32,
        {static_cast<int64_t>(rows), static_cast<int64_t>(width)},
        ids);
  }
  if (config.enable_output_sku_logprobs()) {
    const int32_t width =
        static_cast<int32_t>(output.outputs.front().token_ids_logprobs.size());
    const size_t emitted_rows = convert_items ? items.size() : rows;
    std::vector<float> values;
    values.reserve(emitted_rows * static_cast<size_t>(width));
    for (size_t i = 0; i < emitted_rows; ++i) {
      const size_t output_index =
          convert_items ? static_cast<size_t>(items[i].output_index) : i;
      append_rec_logprobs_to_vector(
          &values, output.outputs[output_index], width);
    }
    entries[count++] = make_float_output_tensor(
        "sku_logprobs", {static_cast<int64_t>(emitted_rows), width}, values);
  }
  if (convert_items && config.enable_extended_item_info()) {
    std::vector<std::string> dids;
    std::vector<std::string> types;
    dids.reserve(items.size());
    types.reserve(items.size());
    for (const auto& item : items) {
      dids.emplace_back(item.item_info ? item.item_info->did : "");
      types.emplace_back(item.item_info ? item.item_info->type : "");
    }
    entries[count++] = make_string_output_tensor("item_did", dids);
    entries[count++] = make_string_output_tensor("item_type", types);
  }
  if (config.max_decode_rounds() == 1 &&
      build_vocab_probs_output_tensor(output, &entries[count])) {
    ++count;
  }
  response->output_tensors.entries_size = count;
  return true;
}

void copy_rec_output_extended_strings(XLLM_RecOutput* rec_output,
                                      const SequenceOutput& seq_output,
                                      size_t copied_item_count) {
  if (rec_output == nullptr || copied_item_count == 0 ||
      !RecConfig::get_instance().enable_extended_item_info()) {
    return;
  }

  rec_output->item_dids = new char*[copied_item_count]();
  rec_output->item_types = new char*[copied_item_count]();
  CHECK(nullptr != rec_output->item_dids);
  CHECK(nullptr != rec_output->item_types);

  if (!seq_output.item_infos_list.empty() &&
      seq_output.item_infos_list.size() == seq_output.item_ids_list.size()) {
    for (size_t j = 0; j < copied_item_count; ++j) {
      const RecItemInfo& info = seq_output.item_infos_list[j];
      rec_output->item_dids[j] = duplicate_c_string(info.did);
      rec_output->item_types[j] = duplicate_c_string(info.type);
    }
    return;
  }

  if (copied_item_count == 1 && seq_output.item_info.has_value()) {
    rec_output->item_dids[0] = duplicate_c_string(seq_output.item_info->did);
    rec_output->item_types[0] = duplicate_c_string(seq_output.item_info->type);
  }
}

bool convert_c_infer_input_tensors_to_onerec_mm_data(
    const XLLM_InferInputTensorDesc* tensors,
    size_t tensor_count,
    xllm::MMData* mm_data,
    std::string* error_info) {
  if (mm_data == nullptr) {
    if (error_info != nullptr) {
      *error_info = "mm_data is null";
    }
    return false;
  }

  *mm_data = xllm::MMData();
  if (tensors == nullptr || tensor_count == 0) {
    if (error_info != nullptr) {
      *error_info = "tensors is null or tensor_count is 0";
    }
    return false;
  }

  xllm::MMDict mm_dict;
  mm_dict.reserve(tensor_count);
  bool has_sparse_embedding = false;

  for (size_t index = 0; index < tensor_count; ++index) {
    const XLLM_InferInputTensorDesc& tensor_desc = tensors[index];
    if (tensor_desc.name == nullptr || tensor_desc.name[0] == '\0') {
      if (error_info != nullptr) {
        *error_info = "tensor has empty or null name";
      }
      return false;
    }
    if (tensor_desc.shape == nullptr || tensor_desc.shape_len != 2) {
      if (error_info != nullptr) {
        *error_info =
            std::string("tensor ") + tensor_desc.name + " must be 2-D";
      }
      return false;
    }
    if (tensor_desc.data == nullptr) {
      if (error_info != nullptr) {
        *error_info = std::string("tensor ") + tensor_desc.name +
                      " has null data pointer";
      }
      return false;
    }
    if (strcmp(tensor_desc.name, kOneRecSparseEmbeddingName) != 0 &&
        strcmp(tensor_desc.name, kOneRecDecoderContextEmbeddingName) != 0) {
      if (error_info != nullptr) {
        *error_info =
            std::string("unsupported OneRec tensor name: ") + tensor_desc.name;
      }
      return false;
    }
    if (mm_dict.find(tensor_desc.name) != mm_dict.end()) {
      if (error_info != nullptr) {
        *error_info =
            std::string("duplicate OneRec tensor name: ") + tensor_desc.name;
      }
      return false;
    }

    const int64_t expected_numel =
        infer_input_tensor_desc_numel(tensor_desc.shape, tensor_desc.shape_len);
    if (expected_numel < 0) {
      if (error_info != nullptr) {
        *error_info =
            std::string("invalid shape for tensor ") + tensor_desc.name;
      }
      return false;
    }
    if (tensor_desc.num_elements != static_cast<size_t>(expected_numel)) {
      if (error_info != nullptr) {
        *error_info =
            std::string("num_elements mismatch for tensor ") + tensor_desc.name;
      }
      return false;
    }

    torch::ScalarType scalar_type;
    switch (static_cast<proto::DataType>(tensor_desc.data_type)) {
      case proto::DataType::FLOAT:
        scalar_type = torch::kFloat32;
        break;
      case proto::DataType::BFLOAT16:
        scalar_type = torch::kBFloat16;
        break;
      case proto::DataType::FLOAT16:
        scalar_type = torch::kFloat16;
        break;
      default:
        if (error_info != nullptr) {
          *error_info =
              std::string("unsupported OneRec tensor data_type for ") +
              tensor_desc.name;
        }
        return false;
    }

    std::vector<int64_t> dims(tensor_desc.shape,
                              tensor_desc.shape + tensor_desc.shape_len);
    torch::Tensor tensor =
        torch::from_blob(const_cast<void*>(tensor_desc.data),
                         dims,
                         torch::TensorOptions().dtype(scalar_type));
    // A timeout can return before the worker finishes. Keep request-owned
    // storage even for BF16 so the caller may immediately reuse its buffer.
    tensor = scalar_type == torch::kBFloat16 ? tensor.clone()
                                             : tensor.to(torch::kBFloat16);

    mm_dict[tensor_desc.name] = tensor;
    if (strcmp(tensor_desc.name, kOneRecSparseEmbeddingName) == 0) {
      has_sparse_embedding = true;
    }
  }

  if (!has_sparse_embedding) {
    if (error_info != nullptr) {
      *error_info = "OneRec input_tensors must include 'sparse_embedding'";
    }
    return false;
  }

  *mm_data = xllm::MMData(xllm::MMType::EMBEDDING, mm_dict);
  return true;
}

}  // namespace xllm::helper

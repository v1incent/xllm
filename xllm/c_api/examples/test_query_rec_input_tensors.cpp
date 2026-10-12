/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "c_api/rec.h"

namespace {
constexpr int32_t kFloat = 1;
constexpr int32_t kInt32 = 6;
constexpr int32_t kInt64 = 7;
constexpr int32_t kString = 8;

struct DemoOptions {
  std::string model;
  std::string device = "npu:0";
  std::string sparse;
  std::string context;
  std::string filter;
  std::string output;
  std::string golden;
  std::vector<int32_t> variant_lengths;
  int64_t hidden = 2048;
  int32_t sequence_length = 8;
  int32_t beam_width = 128;
  int32_t top_k = 128;
  int32_t rounds = 3;
  int32_t requests = 2;
  int32_t threads = 1;
  int32_t workers = 1;
  bool graph = false;
  bool convert_items = false;
};

DemoOptions parse_options(int argc, char** argv) {
  DemoOptions result;
  for (int32_t i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--help") {
      std::cout << "--model DIR [--device npu:0] [--sparse FLOAT32.bin] "
                   "[--context FLOAT32.bin] [--hidden 2048] [--seq-len 8] "
                   "[--filter PATH] [--beam-width 128] [--top-k 128] "
                   "[--rounds 3] [--requests 2] [--threads 1] [--workers 1] "
                   "[--graph 0|1] [--convert-items 0|1] [--output FILE] "
                   "[--golden FILE] [--variant-lengths 8,276,8]\n";
      return result;
    }
    if (++i >= argc) {
      throw std::invalid_argument("Missing value for " + key);
    }
    const std::string value(argv[i]);
    if (key == "--model") {
      result.model = value;
    } else if (key == "--device") {
      result.device = value;
    } else if (key == "--sparse") {
      result.sparse = value;
    } else if (key == "--context") {
      result.context = value;
    } else if (key == "--filter") {
      result.filter = value;
    } else if (key == "--output") {
      result.output = value;
    } else if (key == "--golden") {
      result.golden = value;
    } else if (key == "--hidden") {
      result.hidden = std::stoll(value);
    } else if (key == "--seq-len") {
      result.sequence_length = std::stoi(value);
    } else if (key == "--variant-lengths") {
      std::istringstream lengths(value);
      std::string length;
      result.variant_lengths.reserve(
          1 + std::count(value.begin(), value.end(), ','));
      while (std::getline(lengths, length, ',')) {
        const int32_t parsed = std::stoi(length);
        if (parsed <= 0) {
          throw std::invalid_argument("Variant lengths must be positive");
        }
        result.variant_lengths.emplace_back(parsed);
      }
      if (result.variant_lengths.empty()) {
        throw std::invalid_argument("Specify at least one variant length");
      }
    } else if (key == "--beam-width") {
      result.beam_width = std::stoi(value);
    } else if (key == "--top-k") {
      result.top_k = std::stoi(value);
    } else if (key == "--rounds") {
      result.rounds = std::stoi(value);
    } else if (key == "--requests") {
      result.requests = std::stoi(value);
    } else if (key == "--threads") {
      result.threads = std::stoi(value);
    } else if (key == "--workers") {
      result.workers = std::stoi(value);
    } else if (key == "--graph") {
      result.graph = std::stoi(value) != 0;
    } else if (key == "--convert-items") {
      result.convert_items = std::stoi(value) != 0;
    } else {
      throw std::invalid_argument("Unknown argument " + key);
    }
  }
  if (result.model.empty() || result.hidden <= 0 ||
      result.sequence_length <= 0 || result.beam_width <= 0 ||
      result.top_k <= 0 || result.rounds <= 0 || result.requests <= 0 ||
      result.threads <= 0 || result.workers <= 0) {
    throw std::invalid_argument(
        "Specify --model and positive dimensions/counts");
  }
  if (!result.sparse.empty() && !result.variant_lengths.empty()) {
    throw std::invalid_argument(
        "--variant-lengths generates synthetic inputs; omit --sparse");
  }
  return result;
}

std::vector<float> load_tensor(const std::string& path, int64_t hidden) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open tensor " + path);
  }
  const int64_t size = file.tellg();
  if (size <= 0 || size % (hidden * static_cast<int64_t>(sizeof(float))) != 0) {
    throw std::runtime_error("Tensor must contain complete nonempty FP32 rows");
  }
  std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  if (!file) {
    throw std::runtime_error("Failed reading " + path);
  }
  return data;
}

template <typename Value>
void dump_values(std::ostream& stream, const XLLM_InferOutputTensor& tensor) {
  const auto* data = static_cast<const Value*>(tensor.data);
  for (size_t i = 0; i < tensor.num_elements; ++i) {
    stream << ' ' << data[i];
  }
}

std::string dump_response(const XLLM_Response& response) {
  if (response.status_code != kSuccess) {
    throw std::runtime_error(response.error_info);
  }
  if (response.output_tensors.entries_size == 0) {
    throw std::runtime_error("No raw output tensors");
  }
  std::ostringstream stream;
  stream << std::setprecision(9);
  for (size_t i = 0; i < response.output_tensors.entries_size; ++i) {
    const auto& tensor = response.output_tensors.entries[i];
    stream << tensor.name << ' ' << tensor.data_type;
    size_t elements = 1;
    for (size_t j = 0; j < tensor.shape_len; ++j) {
      if (tensor.shape[j] < 0) {
        throw std::runtime_error("Negative output dimension");
      }
      elements *= static_cast<size_t>(tensor.shape[j]);
      stream << ' ' << tensor.shape[j];
    }
    if (elements != tensor.num_elements ||
        (elements > 0 && tensor.data == nullptr)) {
      throw std::runtime_error("Output shape and buffer disagree");
    }
    stream << " :";
    if (tensor.data_type == kInt32) {
      dump_values<int32_t>(stream, tensor);
    } else if (tensor.data_type == kInt64) {
      dump_values<int64_t>(stream, tensor);
    } else if (tensor.data_type == kFloat) {
      dump_values<float>(stream, tensor);
      if (std::strcmp(tensor.name, "vocab_probs") == 0) {
        if (tensor.shape_len != 2 || tensor.shape[1] <= 0) {
          throw std::runtime_error("Expected [rows, vocab] probabilities");
        }
        const auto* values = static_cast<const float*>(tensor.data);
        for (int64_t row = 0; row < tensor.shape[0]; ++row) {
          double sum = 0.0;
          for (int64_t column = 0; column < tensor.shape[1]; ++column) {
            const float value = values[row * tensor.shape[1] + column];
            if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
              throw std::runtime_error("Invalid vocabulary probability");
            }
            sum += value;
          }
          if (std::abs(sum - 1.0) > 1e-4) {
            throw std::runtime_error(
                "Vocabulary probabilities do not sum to one");
          }
        }
      }
    } else if (tensor.data_type == kString) {
      const auto* cursor = static_cast<const char*>(tensor.data);
      for (size_t index = 0; index < tensor.num_elements; ++index) {
        uint32_t length = 0;
        std::memcpy(&length, cursor, sizeof(length));
        cursor += sizeof(length);
        stream << ' ' << std::quoted(std::string(cursor, length));
        cursor += length;
      }
    } else {
      throw std::runtime_error("Unsupported demo output dtype");
    }
    stream << '\n';
  }
  return stream.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const DemoOptions demo = parse_options(argc, argv);
    if (demo.model.empty()) {
      return 0;
    }
    std::vector<std::vector<float>> sparse_variants;
    if (demo.sparse.empty()) {
      const std::vector<int32_t> lengths =
          demo.variant_lengths.empty()
              ? std::vector<int32_t>{demo.sequence_length}
              : demo.variant_lengths;
      sparse_variants.reserve(lengths.size());
      for (size_t variant = 0; variant < lengths.size(); ++variant) {
        auto& sparse = sparse_variants.emplace_back(
            static_cast<size_t>(lengths[variant] * demo.hidden));
        for (size_t i = 0; i < sparse.size(); ++i) {
          sparse[i] = static_cast<float>(
                          static_cast<int32_t>((i + variant * 7) % 17) - 8) *
                      0.001F;
        }
      }
      std::cerr << "Using synthetic embeddings: smoke/AA validation only\n";
    } else {
      sparse_variants.reserve(1);
      sparse_variants.emplace_back(load_tensor(demo.sparse, demo.hidden));
    }
    const std::vector<float> context =
        demo.context.empty() ? std::vector<float>{}
                             : load_tensor(demo.context, demo.hidden);
    const int64_t context_shape[] = {
        static_cast<int64_t>(context.size()) / demo.hidden, demo.hidden};
    XLLM_InitOptions init;
    xllm_rec_init_options_default(&init);
    init.block_size = 128;
    // The generic REC default is smaller than one Oxygen CrossAttention KV
    // block. Allow bounded encoder pages for the concurrent mixed-length cases.
    init.max_cache_size = 128 * 1024 * 1024;
    init.max_seqs_per_batch = demo.beam_width;
    init.max_tokens_per_batch = 4096;
    init.max_decode_rounds = demo.rounds;
    init.beam_width = demo.beam_width;
    init.rec_worker_max_concurrency = demo.workers;
    init.enable_graph = demo.graph;
    init.enable_onerec_prefill_acl_graph = demo.graph;
    init.enable_constrained_decoding = !demo.filter.empty();
    init.enable_convert_tokens_to_item = demo.convert_items;
    init.enable_output_sku_logprobs = true;
    init.max_memory_utilization = 0.3;
    if (demo.filter.size() >= sizeof(init.constrained_decoding_filter_path)) {
      throw std::runtime_error("Filter path is too long for the C API");
    }
    std::snprintf(init.constrained_decoding_filter_path,
                  sizeof(init.constrained_decoding_filter_path),
                  "%s",
                  demo.filter.c_str());
    std::unique_ptr<XLLM_REC_Handler, decltype(&xllm_rec_destroy)> handler(
        xllm_rec_create(), xllm_rec_destroy);
    if (!xllm_rec_initialize(
            handler.get(), demo.model.c_str(), demo.device.c_str(), &init)) {
      throw std::runtime_error("REC initialization failed");
    }
    auto model_path = std::filesystem::path(demo.model).lexically_normal();
    while (!model_path.has_filename() && model_path != model_path.root_path()) {
      model_path = model_path.parent_path();
    }
    const std::string model_id = model_path.filename().string();
    auto infer = [&](size_t variant, int32_t request_number) {
      const auto& sparse = sparse_variants[variant];
      const int64_t sparse_shape[] = {
          static_cast<int64_t>(sparse.size()) / demo.hidden, demo.hidden};
      const XLLM_InferInputTensorDesc inputs[] = {{"sparse_embedding",
                                                   kFloat,
                                                   sparse_shape,
                                                   2,
                                                   sparse.data(),
                                                   sparse.size()},
                                                  {"decoder_context_embedding",
                                                   kFloat,
                                                   context_shape,
                                                   2,
                                                   context.data(),
                                                   context.size()}};
      XLLM_RequestParams params;
      xllm_rec_request_params_default(&params);
      params.max_tokens = demo.rounds;
      params.beam_width = demo.beam_width;
      // --top-k controls candidates per beam. Keep sampling top_k at -1
      // so temperature=0 retains deterministic beam search semantics.
      params.top_logprobs = demo.top_k;
      params.logprobs = true;
      std::snprintf(params.request_id,
                    sizeof(params.request_id),
                    "migration-demo-%d-%zu",
                    request_number,
                    variant);
      std::unique_ptr<XLLM_Response, decltype(&xllm_rec_free_response)>
          response(xllm_rec_input_tensors_completions(handler.get(),
                                                      model_id.c_str(),
                                                      inputs,
                                                      context.empty() ? 1 : 2,
                                                      120000,
                                                      &params),
                   xllm_rec_free_response);
      if (!response) {
        throw std::runtime_error("Null response");
      }
      XLLM_InferTimingDetail timing{};
      xllm_rec_take_last_infer_timing(response->id, &timing);
      return dump_response(*response);
    };
    std::vector<std::string> references;
    references.reserve(sparse_variants.size());
    std::ostringstream combined;
    for (size_t variant = 0; variant < sparse_variants.size(); ++variant) {
      references.emplace_back(infer(variant, 0));
      if (sparse_variants.size() > 1) {
        combined << "variant " << variant << " sequence_length "
                 << sparse_variants[variant].size() / demo.hidden << '\n';
      }
      combined << references.back();
    }
    const std::string reference = combined.str();
    if (!demo.output.empty()) {
      std::ofstream file(demo.output);
      file << reference;
      if (!file) {
        throw std::runtime_error("Failed to write output file");
      }
    }
    if (!demo.golden.empty()) {
      std::ifstream file(demo.golden);
      if (!file) {
        throw std::runtime_error("Cannot open golden file");
      }
      const std::string golden((std::istreambuf_iterator<char>(file)), {});
      if (golden != reference) {
        throw std::runtime_error("Golden output differs (exact comparison)");
      }
    }
    std::atomic<int32_t> next{1};
    std::atomic<int32_t> failed{0};
    std::vector<std::thread> threads;
    threads.reserve(demo.threads);
    for (int32_t i = 0; i < demo.threads; ++i) {
      threads.emplace_back([&] {
        for (int32_t n = next.fetch_add(1); n <= demo.requests;
             n = next.fetch_add(1)) {
          try {
            const size_t variant =
                static_cast<size_t>(n - 1) % sparse_variants.size();
            const std::string actual = infer(variant, n);
            if (actual != references[variant]) {
              if (!demo.output.empty()) {
                const std::string path =
                    demo.output + ".request-" + std::to_string(n) +
                    "-variant-" + std::to_string(variant) + ".txt";
                std::ofstream difference(path);
                difference << actual;
                if (!difference) {
                  throw std::runtime_error("Failed to save mismatch: " + path);
                }
              }
              std::cerr << "request " << n << " variant " << variant
                        << ": output differs\n";
              ++failed;
            }
          } catch (const std::exception& error) {
            std::cerr << "request " << n << ": " << error.what() << '\n';
            ++failed;
          }
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    std::cout << "requests=" << demo.requests
              << " mismatches_or_errors=" << failed << '\n';
    return failed == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

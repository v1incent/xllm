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

#pragma once

#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace xllm {
class Tokenizer;
struct SequenceOutput;

// Keep the first metadata entry for each item. Shuffle only the selected
// prefix, so sampling work is bounded by the configured conversion limit.
inline std::vector<size_t> select_unique_rec_item_indices(
    const std::vector<int64_t>& item_ids,
    int32_t limit,
    uint32_t seed) {
  std::vector<size_t> indices;
  indices.reserve(item_ids.size());
  std::unordered_set<int64_t> seen;
  seen.reserve(item_ids.size());
  for (size_t i = 0; i < item_ids.size(); ++i) {
    if (seen.insert(item_ids[i]).second) {
      indices.emplace_back(i);
    }
  }
  if (limit <= 0 || indices.size() <= static_cast<size_t>(limit)) {
    return indices;
  }
  std::mt19937 generator(seed);
  for (size_t i = 0; i < static_cast<size_t>(limit); ++i) {
    std::uniform_int_distribution<size_t> distribution(i, indices.size() - 1);
    std::swap(indices[i], indices[distribution(generator)]);
  }
  indices.resize(static_cast<size_t>(limit));
  return indices;
}

void decode_rec_items(const Tokenizer& tokenizer,
                      bool skip_special_tokens,
                      size_t sequence_index,
                      SequenceOutput& output);
}  // namespace xllm

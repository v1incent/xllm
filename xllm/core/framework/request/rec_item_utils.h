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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core/util/rec_item_sampling.h"

namespace xllm {
class Tokenizer;
struct SequenceOutput;

// Sample raw entries first, then keep the first sampled metadata entry for
// each item, matching v0.9. Duplicates may reduce the result below the limit.
inline std::vector<size_t> select_unique_rec_item_indices(
    const std::vector<int64_t>& item_ids,
    int32_t limit,
    int32_t random_seed,
    size_t sequence_index = 0) {
  const size_t count =
      limit <= 0 ? item_ids.size()
                 : std::min(item_ids.size(), static_cast<size_t>(limit));
  std::vector<size_t> indices;
  indices.reserve(count);
  std::unordered_set<int64_t> seen;
  seen.reserve(count);
  for_each_sampled_rec_item_index(
      item_ids.size(), limit, random_seed, sequence_index, [&](size_t index) {
        if (seen.insert(item_ids[index]).second) {
          indices.emplace_back(index);
        }
      });
  return indices;
}

void decode_rec_items(const Tokenizer& tokenizer,
                      bool skip_special_tokens,
                      size_t sequence_index,
                      SequenceOutput& output);
}  // namespace xllm

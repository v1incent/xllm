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
#include <random>
#include <unordered_set>
#include <vector>

namespace xllm {

// Preserve the v0.9 entry sampling order, before item-ID deduplication. Visit
// distinct indices without constructing or shuffling a full-size index array.
// A nonpositive limit visits everything in source order; limit=1 is O(1).
// Larger limits retain v0.9 rejection sampling, which slows as limit nears
// size.
template <typename Emit>
inline void for_each_sampled_rec_item_index(size_t size,
                                            int32_t limit,
                                            int32_t random_seed,
                                            size_t sequence_index,
                                            Emit emit) {
  if (size == 0) {
    return;
  }
  if (limit <= 0 || static_cast<size_t>(limit) >= size) {
    for (size_t index = 0; index < size; ++index) {
      emit(index);
    }
    return;
  }

  std::mt19937 seeded_rng;
  std::mt19937* generator = nullptr;
  if (random_seed >= 0) {
    seeded_rng = std::mt19937(static_cast<uint32_t>(random_seed) +
                              static_cast<uint32_t>(sequence_index));
    generator = &seeded_rng;
  } else {
    static thread_local std::mt19937 rng{std::random_device{}()};
    generator = &rng;
  }

  const size_t count = static_cast<size_t>(limit);
  std::uniform_int_distribution<size_t> distribution(0, size - 1);
  if (count == 1) {
    emit(distribution(*generator));
    return;
  }

  std::unordered_set<size_t> picked;
  picked.reserve(count * 2);
  while (picked.size() < count) {
    picked.insert(distribution(*generator));
  }
  for (size_t index : picked) {
    emit(index);
  }
}

// Copy only selected entries from a dictionary's existing vector. Append to
// the caller's output, keeping each item's ID and metadata together.
template <typename T>
inline void append_sampled_rec_items(const std::vector<T>& src,
                                     int32_t limit,
                                     int32_t random_seed,
                                     size_t sequence_index,
                                     std::vector<T>* out) {
  if (out == nullptr || src.empty()) {
    return;
  }
  const size_t count = limit <= 0
                           ? src.size()
                           : std::min(src.size(), static_cast<size_t>(limit));
  out->reserve(out->size() + count);
  for_each_sampled_rec_item_index(
      src.size(), limit, random_seed, sequence_index, [&](size_t index) {
        out->emplace_back(src[index]);
      });
}

}  // namespace xllm

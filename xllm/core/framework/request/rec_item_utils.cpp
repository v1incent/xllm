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

#include "core/framework/request/rec_item_utils.h"

#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/request/request_output.h"
#include "core/framework/tokenizer/rec_tokenizer.h"
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {
void decode_rec_items(const Tokenizer& tokenizer,
                      bool skip_special_tokens,
                      size_t sequence_index,
                      SequenceOutput& output) {
  const auto& config = RecConfig::get_instance();
  constexpr size_t kItemTokenCount = static_cast<size_t>(REC_TOKEN_SIZE);
  if (!config.enable_convert_tokens_to_item() || output.token_ids.empty() ||
      output.token_ids.size() % kItemTokenCount != 0) {
    return;
  }
  const bool extended = config.enable_extended_item_info();
  const auto* rec_tokenizer = dynamic_cast<const RecTokenizer*>(&tokenizer);
  std::vector<RecItemInfo> candidates;
  candidates.reserve(output.token_ids.size() / kItemTokenCount);
  for (size_t offset = 0; offset < output.token_ids.size();
       offset += kItemTokenCount) {
    const Slice<int32_t> tokens(output.token_ids.data() + offset,
                                kItemTokenCount);
    std::vector<RecItemInfo> infos;
    if (extended && rec_tokenizer != nullptr &&
        rec_tokenizer->decode_item_infos(tokens, &infos) && !infos.empty()) {
      candidates.insert(candidates.end(), infos.begin(), infos.end());
      continue;
    }
    // Older dictionaries/tokenizers may have IDs without extended metadata.
    std::vector<int64_t> ids;
    if (!tokenizer.decode(tokens, skip_special_tokens, &ids)) {
      continue;
    }
    candidates.reserve(candidates.size() + ids.size());
    for (int64_t id : ids) {
      candidates.emplace_back(RecItemInfo{id, "", ""});
    }
  }
  std::vector<int64_t> ids;
  ids.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    ids.emplace_back(candidate.item_id);
  }
  const int32_t random_seed = ExecutionConfig::get_instance().random_seed();
  const uint32_t seed = random_seed >= 0
                            ? static_cast<uint32_t>(random_seed) +
                                  static_cast<uint32_t>(sequence_index)
                            : std::random_device{}();
  const auto indices = select_unique_rec_item_indices(
      ids, config.each_conversion_threshold(), seed);
  output.item_ids_list.reserve(indices.size());
  if (extended) {
    output.item_infos_list.reserve(indices.size());
  }
  for (size_t index : indices) {
    output.item_ids_list.emplace_back(candidates[index].item_id);
    if (extended) {
      output.item_infos_list.emplace_back(std::move(candidates[index]));
    }
  }
  if (!output.item_ids_list.empty()) {
    output.item_ids = output.item_ids_list.front();
    if (extended) {
      output.item_info = output.item_infos_list.front();
    }
  }
}
}  // namespace xllm

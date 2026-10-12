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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::detail {

// Select BW children from BW * K candidates, grouped by parent for cache copy.
// All tensor operations stay on the input device and current stream.
inline void select_rec_beams(const torch::Tensor& logprobs,
                             const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             const torch::Tensor& sequence_group,
                             int64_t current_step,
                             torch::Tensor& out_token_ids,
                             torch::Tensor& out_token_index,
                             torch::Tensor& out_log_probs,
                             torch::Tensor& out_beam_count_prefix_sums,
                             torch::Tensor& out_sequence) {
  CHECK_EQ(sequence_group.dim(), 3);
  CHECK_EQ(top_tokens.dim(), 2);
  CHECK_EQ(top_tokens.sizes(), top_logprobs.sizes());
  CHECK_GT(current_step, 0);
  CHECK_LT(current_step, sequence_group.size(2));
  const int64_t batch_size = sequence_group.size(0);
  const int64_t beam_width = sequence_group.size(1);
  const int64_t rounds = sequence_group.size(2);
  const int64_t candidates = top_tokens.size(1);
  CHECK_GT(batch_size, 0);
  CHECK_GT(beam_width, 0);
  CHECK_GT(candidates, 0);
  CHECK_EQ(top_tokens.size(0), batch_size * beam_width);
  CHECK_EQ(logprobs.numel(), batch_size * beam_width);
  CHECK_EQ(out_sequence.sizes(), sequence_group.sizes());
  CHECK_EQ(out_token_ids.numel(), batch_size * beam_width);
  CHECK_EQ(out_token_index.numel(), batch_size * beam_width);
  CHECK_EQ(out_log_probs.numel(), batch_size * beam_width);
  CHECK_EQ(out_beam_count_prefix_sums.numel(), batch_size * beam_width);

  const auto scores =
      (logprobs.reshape({batch_size, beam_width, 1}) +
       top_logprobs.reshape({batch_size, beam_width, candidates}))
          .reshape({batch_size, beam_width * candidates});
  auto [selected_scores, selected_indices] =
      scores.topk(beam_width, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  const auto index_options = top_tokens.options().dtype(torch::kInt64);
  const auto ranks = torch::arange(beam_width, index_options).unsqueeze(0);
  // Include score rank in the key to preserve descending scores per parent.
  const auto keys =
      torch::floor_divide(selected_indices, candidates) * beam_width + ranks;
  const auto order =
      torch::argsort(keys, /*stable=*/true, /*dim=*/1, /*descending=*/false);
  selected_indices = selected_indices.gather(1, order);
  selected_scores = selected_scores.gather(1, order);
  const auto parents = torch::floor_divide(selected_indices, candidates);
  const auto batch_offsets =
      torch::arange(batch_size, index_options).unsqueeze(1) * beam_width;
  const auto global_parents = parents + batch_offsets;
  const auto tokens = top_tokens.reshape({batch_size, beam_width * candidates})
                          .gather(1, selected_indices);

  out_token_ids.reshape({batch_size, beam_width}).copy_(tokens);
  out_token_index.reshape({batch_size, beam_width}).copy_(global_parents);
  out_log_probs.reshape({batch_size, beam_width}).copy_(selected_scores);
  const auto counts = parents.unsqueeze(1)
                          .eq(ranks.unsqueeze(2))
                          .sum(/*dim=*/2, /*keepdim=*/false, torch::kInt32);
  out_beam_count_prefix_sums.reshape({batch_size, beam_width})
      .copy_(counts.cumsum(/*dim=*/1) + batch_offsets);
  out_sequence.slice(2, 0, current_step)
      .copy_(sequence_group.reshape({-1, rounds})
                 .index_select(0, global_parents.reshape({-1}))
                 .reshape({batch_size, beam_width, rounds})
                 .slice(2, 0, current_step));
  out_sequence.select(2, current_step).copy_(tokens);
}

}  // namespace xllm::kernel::detail

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "core/platform/npu/python_runtime.h"

namespace xllm::kernel::npu {
namespace {

void check_beams(int64_t beam_width, int64_t candidates, int64_t step) {
  constexpr int64_t kBatchSize = 2;
  const int64_t rounds = step + 2;
  const int64_t rows = kBatchSize * beam_width;
  const int64_t count = rows * candidates;
  const auto options = torch::TensorOptions().dtype(torch::kInt32);
  const auto tokens =
      torch::arange(1000, 1000 + count, options).reshape({rows, candidates});
  const auto history = torch::arange(rows * rounds, options)
                           .reshape({kBatchSize, beam_width, rounds});
  const auto logprobs =
      -torch::arange(rows, torch::kFloat32).reshape({rows, 1});
  auto probs = torch::empty({rows, candidates}, torch::kFloat32);
  float* data = probs.data_ptr<float>();
  for (int64_t i = 0; i < count; ++i) {
    // Unique deterministic totals, with winners spread across parent beams.
    data[i] = static_cast<float>((i * 37) % count) / 16.0F +
              static_cast<float>(i / candidates);
  }
  auto expected_tokens = torch::empty({rows, 1}, options);
  auto expected_parents = torch::empty({rows, 1}, options);
  auto expected_scores = torch::empty({rows, 1}, torch::kFloat32);
  auto expected_prefix = torch::empty({rows, 1}, options);
  auto expected_history = torch::full_like(history, -99);
  const auto totals = (probs + logprobs).contiguous();
  const float* scores = totals.data_ptr<float>();
  for (int64_t batch = 0; batch < kBatchSize; ++batch) {
    std::vector<int64_t> indices(beam_width * candidates);
    std::iota(indices.begin(), indices.end(), batch * beam_width * candidates);
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
      return scores[a] > scores[b];
    });
    indices.resize(beam_width);
    std::stable_sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
      return a / candidates < b / candidates;
    });
    std::vector<int32_t> counts(beam_width, 0);
    for (int64_t i = 0; i < beam_width; ++i) {
      const int64_t row = batch * beam_width + i;
      const int64_t selected = indices[i];
      const int64_t parent = selected / candidates;
      expected_tokens[row][0] = 1000 + selected;
      expected_parents[row][0] = parent;
      expected_scores[row][0] = scores[selected];
      ++counts[parent - batch * beam_width];
      expected_history[batch][i]
          .slice(0, 0, step)
          .copy_(history.reshape({rows, rounds})[parent].slice(0, 0, step));
      expected_history[batch][i][step] = 1000 + selected;
    }
    int64_t prefix = batch * beam_width;
    for (int64_t i = 0; i < beam_width; ++i) {
      prefix += counts[i];
      expected_prefix[batch * beam_width + i][0] = prefix;
    }
  }

  const torch::Device device(torch::kPrivateUse1, 0);
  auto device_history = history.to(device);
  auto out_tokens = torch::full({rows, 1}, -99, options.device(device));
  auto out_parents = torch::empty_like(out_tokens);
  auto out_scores = torch::empty({rows, 1}, logprobs.options().device(device));
  auto out_prefix = torch::empty_like(out_tokens);
  auto out_history = torch::full_like(device_history, -99);
  beam_search_rec(logprobs.to(device),
                  tokens.to(device),
                  probs.to(device),
                  device_history,
                  step,
                  out_tokens,
                  out_parents,
                  out_scores,
                  out_prefix,
                  out_history);
  EXPECT_TRUE(torch::equal(out_tokens.cpu(), expected_tokens));
  EXPECT_TRUE(torch::equal(out_parents.cpu(), expected_parents));
  EXPECT_TRUE(torch::equal(out_scores.cpu(), expected_scores));
  EXPECT_TRUE(torch::equal(out_prefix.cpu(), expected_prefix));
  EXPECT_TRUE(torch::equal(out_history.cpu(), expected_history));
}

TEST(NpuRecBeamSearchTest, PreservesTokensParentsAndCacheMetadata) {
  ASSERT_TRUE(xllm::npu::init_python_runtime(/*device_index=*/0));
  torch_npu::init_npu(static_cast<int8_t>(0));
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {1, 1}, {3, 5}, {8, 8}, {16, 32}, {16, 3}, {128, 128}}) {
    for (int64_t step : {1, 2, 8}) {
      SCOPED_TRACE(::testing::Message() << "BW=" << shape[0] << " K="
                                        << shape[1] << " step=" << step);
      check_beams(shape[0], shape[1], step);
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::npu

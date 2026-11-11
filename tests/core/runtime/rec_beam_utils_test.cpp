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

#include "core/runtime/rec_beam_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>

namespace xllm::runtime::detail {
namespace {
TEST(RecBeamUtilsTest, ConstraintMasksFollowEveryBeamAndExcludeFutureTokens) {
  const auto histories =
      torch::tensor({11, 21, 901, 11, 22, 902, 31, 41, 903, 32, 42, 904},
                    torch::kInt32)
          .reshape({2, 2, 3});
  EXPECT_EQ(beam_constraint_prefixes(histories, 0),
            (std::vector<std::vector<int32_t>>{{}, {}}));
  EXPECT_EQ(beam_constraint_prefixes(histories, 1),
            (std::vector<std::vector<int32_t>>{{11}, {11}, {31}, {32}}));
  EXPECT_EQ(beam_constraint_prefixes(histories, 2),
            (std::vector<std::vector<int32_t>>{
                {11, 21}, {11, 22}, {31, 41}, {32, 42}}));
}

TEST(RecBeamUtilsTest, FirstRoundKeepsRequestsInSeparateBeamRows) {
  auto tokens =
      torch::tensor({11, 12, 13, 21, 22, 23}, torch::kInt32).reshape({6, 1});
  auto scores =
      torch::tensor({-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}).reshape({6, 1});
  auto output_tokens = torch::zeros_like(tokens);
  auto output_scores = torch::zeros_like(scores);
  auto history = torch::full({2, 3, 3}, -1, torch::kInt32);
  write_first_round_beam_outputs(tokens,
                                 scores,
                                 /*batch_size=*/2,
                                 output_tokens,
                                 output_scores,
                                 history);
  const auto expected = torch::tensor({11,
                                       -1,
                                       -1,
                                       12,
                                       -1,
                                       -1,
                                       13,
                                       -1,
                                       -1,
                                       21,
                                       -1,
                                       -1,
                                       22,
                                       -1,
                                       -1,
                                       23,
                                       -1,
                                       -1},
                                      torch::kInt32)
                            .reshape({2, 3, 3});
  EXPECT_TRUE(torch::equal(output_tokens, tokens));
  EXPECT_TRUE(torch::equal(output_scores, scores));
  EXPECT_TRUE(torch::equal(history, expected));
}

TEST(RecBeamUtilsTest, CumulativeFollowsParentsAcrossWideningAndNarrowing) {
  auto history = torch::zeros({2, 2, 4});
  const auto scores0 = torch::tensor({-1.f, -2.f, -3.f, -4.f}).reshape({4, 1});
  write_beam_round_step_logprobs(
      scores0, {}, /*batch_size=*/2, /*round=*/0, history);
  const auto scores1 =
      torch::tensor({-2.5f, -3.f, -4.5f, -5.f}).reshape({4, 1});
  const auto parents1 = torch::tensor({1, 1, 2, 3}, torch::kInt32);
  write_beam_round_step_logprobs(
      scores1, parents1, /*batch_size=*/2, /*round=*/1, history);
  const auto scores2 =
      torch::tensor({-3.1f, -2.7f, -3.2f, -5.5f, -4.75f, -6.f}).reshape({6, 1});
  const auto parents2 = torch::tensor({1, 0, 1, 3, 2, 3}, torch::kInt32);
  write_beam_round_step_logprobs(
      scores2, parents2, /*batch_size=*/2, /*round=*/2, history);
  const auto expected2 =
      torch::tensor({-2.f, -3.f,  -3.1f,  0.f, -2.f, -2.5f, -2.7f, 0.f,
                     -2.f, -3.f,  -3.2f,  0.f, -4.f, -5.f,  -5.5f, 0.f,
                     -3.f, -4.5f, -4.75f, 0.f, -4.f, -5.f,  -6.f,  0.f})
          .reshape({2, 3, 4});
  EXPECT_TRUE(torch::equal(history, expected2));
  EXPECT_TRUE(torch::equal(history.select(2, 2).reshape({-1, 1}), scores2));

  const auto scores3 = torch::tensor({-3.f, -5.f}).reshape({2, 1});
  const auto parents3 = torch::tensor({1, 4}, torch::kInt32);
  write_beam_round_step_logprobs(
      scores3, parents3, /*batch_size=*/2, /*round=*/3, history);
  const auto expected3 =
      torch::tensor({-2.f, -2.5f, -2.7f, -3.f, -3.f, -4.5f, -4.75f, -5.f})
          .reshape({2, 1, 4});
  EXPECT_TRUE(torch::equal(history, expected3));
  EXPECT_TRUE(torch::equal(history.select(2, 3).reshape({-1, 1}), scores3));
}

TEST(RecBeamUtilsTest, CumulativePreservesScoresLostBySubtractThenAdd) {
  auto history = torch::zeros({1, 1, 2});
  const auto previous = torch::tensor({-2.f}).reshape({1, 1});
  const auto current = torch::tensor({-33554436.f}).reshape({1, 1});
  const auto parents = torch::tensor({0}, torch::kInt32);
  // At this FP32 boundary, subtracting the parent then adding it loses an ULP.
  EXPECT_FALSE(torch::equal((current - previous) + previous, current));
  write_beam_round_step_logprobs(
      previous, {}, /*batch_size=*/1, /*round=*/0, history);
  write_beam_round_step_logprobs(
      current, parents, /*batch_size=*/1, /*round=*/1, history);
  EXPECT_TRUE(torch::equal(history.select(2, 1), current));
}

TEST(RecBeamUtilsTest, HandlesWidthOneAndDisabledCollection) {
  torch::Tensor disabled;
  write_beam_round_step_logprobs(
      {}, {}, /*batch_size=*/1, /*round=*/0, disabled);
  EXPECT_FALSE(disabled.defined());
  auto history = torch::zeros({1, 1, 1});
  auto scores = torch::tensor({-.25f}).reshape({1, 1});
  write_beam_round_step_logprobs(
      scores, {}, /*batch_size=*/1, /*round=*/0, history);
  EXPECT_FLOAT_EQ(history[0][0][0].item<float>(), -.25f);
}

TEST(RecBeamUtilsTest, MaskedParentsKeepImpossibleScoresWithoutNaN) {
  const float invalid = -std::numeric_limits<float>::infinity();
  auto parents = torch::tensor({0, 1}, torch::kInt32);
  auto previous = torch::tensor({-1.f, invalid}).reshape({2, 1});
  auto current = torch::tensor({-1.5f, invalid}).reshape({2, 1});
  auto history = torch::zeros({1, 2, 2});
  write_beam_round_step_logprobs(
      previous, {}, /*batch_size=*/1, /*round=*/0, history);
  write_beam_round_step_logprobs(
      current, parents, /*batch_size=*/1, /*round=*/1, history);
  EXPECT_FALSE(torch::isnan(history).any().item<bool>());
  EXPECT_FLOAT_EQ(history[0][0][1].item<float>(), -1.5f);
  EXPECT_EQ(history[0][1][0].item<float>(), invalid);
  EXPECT_EQ(history[0][1][1].item<float>(), invalid);
}
}  // namespace
}  // namespace xllm::runtime::detail

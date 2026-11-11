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

#include <limits>
#include <memory>

#include "c_api/internal/helper.h"
#include "core/framework/config/rec_config.h"
#include "core/runtime/rec_beam_utils.h"
#include "rec.pb.h"

namespace xllm::helper {
namespace {
using ResponsePtr =
    std::unique_ptr<XLLM_Response, decltype(&xllm_free_response)>;

TEST(RecTensorTest, CopiesBfloatInputBeforeReturningToCaller) {
  std::vector<torch::BFloat16> values(8, torch::BFloat16(1.0F));
  const int64_t shape[] = {2, 4};
  const XLLM_InferInputTensorDesc input{
      "sparse_embedding",
      static_cast<int32_t>(proto::DataType::BFLOAT16),
      shape,
      2,
      values.data(),
      values.size()};
  MMData converted;
  std::string error;
  ASSERT_TRUE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error))
      << error;
  std::fill(values.begin(), values.end(), torch::BFloat16(9.0F));
  const auto& tensor =
      std::get<torch::Tensor>(converted.items<MMDict>().at("sparse_embedding"));
  EXPECT_EQ(tensor.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(tensor.sum().item<float>(), 8.0F);
}

TEST(RecTensorTest, ConvertsFloatAndRejectsMalformedDescriptors) {
  std::vector<float> values(8, 0.5F);
  int64_t shape[] = {2, 4};
  XLLM_InferInputTensorDesc input{"sparse_embedding",
                                  static_cast<int32_t>(proto::DataType::FLOAT),
                                  shape,
                                  2,
                                  values.data(),
                                  values.size()};
  MMData converted;
  std::string error;
  ASSERT_TRUE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
  const auto& tensor =
      std::get<torch::Tensor>(converted.items<MMDict>().at("sparse_embedding"));
  EXPECT_EQ(tensor.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(tensor.sum().item<float>(), 4.0F);
  input.num_elements = 7;
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
  input.num_elements = 8;
  shape[0] = std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
  shape[0] = 2;
  input.name = "unsupported";
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
  input.name = "decoder_context_embedding";
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
  input.name = "sparse_embedding";
  const XLLM_InferInputTensorDesc duplicate[] = {input, input};
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      duplicate, 2, &converted, &error));
  input.data = nullptr;
  EXPECT_FALSE(convert_c_infer_input_tensors_to_onerec_mm_data(
      &input, 1, &converted, &error));
}

class RecOutputTensorTest : public ::testing::Test {
 protected:
  void SetUp() override { saved_ = RecConfig::get_instance(); }
  void TearDown() override { RecConfig::get_instance() = saved_; }
  RecConfig saved_;
};

TEST_F(RecOutputTensorTest, TokenTensorMatchesActualRows) {
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(false)
      .enable_output_sku_logprobs(false)
      .max_decode_rounds(3);
  RequestOutput output;
  output.outputs.resize(2);
  output.outputs[0].token_ids = {1, 2, 3};
  output.outputs[1].token_ids = {4, 5, 6};
  ResponsePtr response(new XLLM_Response{}, xllm_free_response);
  ASSERT_TRUE(populate_raw_output_tensors(
      InferenceType::REC_COMPLETIONS, output, response.get()));
  const auto& tensor = response->output_tensors.entries[0];
  EXPECT_STREQ(tensor.name, "rec_result");
  EXPECT_EQ(tensor.shape[0], 2);
  EXPECT_EQ(tensor.shape[1], 3);
  EXPECT_EQ(tensor.num_elements, 6);
  EXPECT_EQ(static_cast<const int32_t*>(tensor.data)[5], 6);
}

TEST_F(RecOutputTensorTest, ItemCountAndExtendedOutputsStayAligned) {
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(true)
      .enable_output_sku_logprobs(true)
      .enable_extended_item_info(true)
      .total_conversion_threshold(1)
      .max_decode_rounds(3);
  RequestOutput output;
  output.outputs.resize(1);
  output.outputs[0].item_ids_list = {42, 43};
  output.outputs[0].token_ids_logprobs = {-0.5F, -0.8F, -1.0F};
  ResponsePtr response(new XLLM_Response{}, xllm_free_response);
  ASSERT_TRUE(populate_raw_output_tensors(
      InferenceType::REC_COMPLETIONS, output, response.get()));
  ASSERT_EQ(response->output_tensors.entries_size, 4);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(response->output_tensors.entries[i].shape[0], 1);
  }
  EXPECT_EQ(
      static_cast<const int64_t*>(response->output_tensors.entries[0].data)[0],
      42);
}

TEST_F(RecOutputTensorTest, EmptyItemSetHasZeroSizedTensor) {
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(true)
      .enable_output_sku_logprobs(false)
      .enable_extended_item_info(false)
      .max_decode_rounds(3);
  RequestOutput output;
  output.outputs.resize(1);
  ResponsePtr response(new XLLM_Response{}, xllm_free_response);
  ASSERT_TRUE(populate_raw_output_tensors(
      InferenceType::REC_COMPLETIONS, output, response.get()));
  EXPECT_EQ(response->output_tensors.entries[0].shape[0], 0);
  EXPECT_EQ(response->output_tensors.entries[0].num_elements, 0);
}

TEST_F(RecOutputTensorTest, CumulativeBeamScoresReachRawAndStructuredOutputs) {
  RecConfig::get_instance()
      .enable_output_sku_logprobs(true)
      .enable_extended_item_info(false)
      .total_conversion_threshold(10)
      .max_decode_rounds(3);
  auto history = torch::zeros({1, 2, 3});
  const auto scores0 = torch::tensor({-1.f, -2.f}).reshape({2, 1});
  const auto scores1 = torch::tensor({-2.5f, -3.f}).reshape({2, 1});
  const auto scores2 = torch::tensor({-3.25f, -3.5f}).reshape({2, 1});
  const auto parents = torch::tensor({1, 0}, torch::kInt32);
  runtime::detail::write_beam_round_step_logprobs(
      scores0, {}, /*batch_size=*/1, /*round=*/0, history);
  runtime::detail::write_beam_round_step_logprobs(scores1,
                                                  parents,
                                                  /*batch_size=*/1,
                                                  /*round=*/1,
                                                  history);
  runtime::detail::write_beam_round_step_logprobs(scores2,
                                                  parents,
                                                  /*batch_size=*/1,
                                                  /*round=*/2,
                                                  history);
  RequestOutput output;
  output.outputs.resize(2);
  output.outputs[0].token_ids = {11, 12, 13};
  output.outputs[1].token_ids = {21, 22, 23};
  output.outputs[0].item_ids_list = {42, 43};
  output.outputs[1].item_ids_list = {44};
  const auto values = history.accessor<float, 3>();
  for (int32_t row = 0; row < 2; ++row) {
    output.outputs[row].token_ids_logprobs.reserve(3);
    for (int32_t column = 0; column < 3; ++column) {
      output.outputs[row].token_ids_logprobs.emplace_back(
          values[0][row][column]);
    }
  }
  const std::vector<float> expected = {-1.f, -3.f, -3.25f, -2.f, -2.5f, -3.5f};
  for (bool convert_items : {false, true}) {
    SCOPED_TRACE(convert_items);
    RecConfig::get_instance().enable_convert_tokens_to_item(convert_items);
    ResponsePtr response(
        build_success_response(InferenceType::REC_COMPLETIONS,
                               output,
                               RecPipelineType::kOneRecXAttentionPipeline,
                               "score-contract",
                               /*created_time=*/0,
                               "model"),
        xllm_free_response);
    ASSERT_NE(response, nullptr);
    ASSERT_EQ(response->output_tensors.entries_size, 2);
    const auto& raw = response->output_tensors.entries[1];
    EXPECT_STREQ(raw.name, "sku_logprobs");
    EXPECT_EQ(raw.data_type, static_cast<int32_t>(proto::DataType::FLOAT));
    ASSERT_EQ(raw.shape_len, 2);
    EXPECT_EQ(raw.shape[0], convert_items ? 3 : 2);
    EXPECT_EQ(raw.shape[1], 3);
    const std::vector<size_t> output_rows = convert_items
                                                ? std::vector<size_t>{0, 0, 1}
                                                : std::vector<size_t>{0, 1};
    ASSERT_EQ(raw.num_elements, output_rows.size() * 3);
    const auto* raw_values = static_cast<const float*>(raw.data);
    for (size_t row = 0; row < output_rows.size(); ++row) {
      for (size_t column = 0; column < 3; ++column) {
        EXPECT_EQ(raw_values[row * 3 + column],
                  expected[output_rows[row] * 3 + column]);
      }
    }
    ASSERT_EQ(response->rec_outputs.entries_size, 2);
    for (size_t row = 0; row < 2; ++row) {
      const auto& structured = response->rec_outputs.entries[row];
      ASSERT_EQ(structured.rec_token_logprobs_size, 3);
      for (size_t column = 0; column < 3; ++column) {
        EXPECT_EQ(structured.rec_token_logprobs[column],
                  expected[row * 3 + column]);
      }
    }
  }
}

TEST_F(RecOutputTensorTest, VocabularyProbabilitiesPreserveRowsAndWidth) {
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(false)
      .enable_output_sku_logprobs(false)
      .max_decode_rounds(1);
  RequestOutput output;
  output.outputs.resize(2);
  output.outputs[0].embedding = torch::tensor({0.2F, 0.3F, 0.5F});
  output.outputs[1].embedding = torch::tensor({0.1F, 0.2F, 0.7F});
  ResponsePtr response(new XLLM_Response{}, xllm_free_response);
  ASSERT_TRUE(populate_raw_output_tensors(
      InferenceType::REC_COMPLETIONS, output, response.get()));
  ASSERT_EQ(response->output_tensors.entries_size, 2);
  const auto& tensor = response->output_tensors.entries[1];
  EXPECT_STREQ(tensor.name, "vocab_probs");
  EXPECT_EQ(tensor.shape[0], 2);
  EXPECT_EQ(tensor.shape[1], 3);
  EXPECT_EQ(tensor.num_elements, 6);
  EXPECT_FLOAT_EQ(static_cast<const float*>(tensor.data)[5], 0.7F);
}

TEST_F(RecOutputTensorTest, RejectsInconsistentVocabularyRows) {
  RecConfig::get_instance()
      .enable_convert_tokens_to_item(false)
      .enable_output_sku_logprobs(false)
      .max_decode_rounds(1);
  RequestOutput output;
  output.outputs.resize(2);
  output.outputs[0].embedding = torch::tensor({0.2F, 0.3F, 0.5F});
  output.outputs[1].embedding = torch::tensor({0.1F, 0.9F});
  ResponsePtr response(new XLLM_Response{}, xllm_free_response);
  EXPECT_THROW(populate_raw_output_tensors(
                   InferenceType::REC_COMPLETIONS, output, response.get()),
               std::invalid_argument);
}
}  // namespace
}  // namespace xllm::helper

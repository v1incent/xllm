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

#include <cstdio>
#include <memory>

#include "c_api/rec.h"

namespace {

using HandlerPtr =
    std::unique_ptr<XLLM_REC_Handler, decltype(&xllm_rec_destroy)>;
using ResponsePtr =
    std::unique_ptr<XLLM_Response, decltype(&xllm_rec_free_response)>;

TEST(RecCApiTest, InitializesDefaultsAndDestroysAnUninitializedHandler) {
  XLLM_InitOptions options{};
  xllm_rec_init_options_default(&options);
  EXPECT_GT(options.beam_width, 0);
  EXPECT_GT(options.max_decode_rounds, 0);
  EXPECT_GT(options.rec_worker_max_concurrency, 0);
  EXPECT_STREQ(options.task, "generate");
  XLLM_RequestParams params{};
  xllm_rec_request_params_default(&params);
  EXPECT_GT(params.n, 0);
  EXPECT_GT(params.max_tokens, 0);
  HandlerPtr handler(xllm_rec_create(), xllm_rec_destroy);
  ASSERT_NE(handler, nullptr);
  xllm_rec_destroy(nullptr);
  xllm_rec_free_response(nullptr);
}

TEST(RecCApiTest, RejectsInvalidTensorCallsAndPreservesRequestId) {
  XLLM_RequestParams params{};
  xllm_rec_request_params_default(&params);
  std::snprintf(params.request_id, sizeof(params.request_id), "%s", "invalid");
  ResponsePtr response(xllm_rec_input_tensors_completions(
                           nullptr, "model", nullptr, 0, 0, &params),
                       xllm_rec_free_response);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status_code, kInvalidRequest);
  EXPECT_STREQ(response->id, "invalid");
  EXPECT_EQ(response->output_tensors.entries_size, 0);
  EXPECT_FALSE(xllm_rec_initialize(nullptr, "model", nullptr));
  EXPECT_FALSE(
      xllm_rec_initialize_with_devices(nullptr, "model", "npu:0", nullptr));
}

TEST(RecCApiTest, RejectsInferenceBeforeModelInitialization) {
  HandlerPtr handler(xllm_rec_create(), xllm_rec_destroy);
  ASSERT_NE(handler, nullptr);
  const int64_t shape[] = {1, 2};
  const float values[] = {0.1F, 0.2F};
  const XLLM_InferInputTensorDesc input{
      "sparse_embedding", /*data_type=*/1, shape, 2, values, 2};
  ResponsePtr response(xllm_rec_input_tensors_completions(
                           handler.get(), "model", &input, 1, 0, nullptr),
                       xllm_rec_free_response);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status_code, kNotInitialized);
  XLLM_InferTimingDetail timing{};
  EXPECT_FALSE(xllm_rec_take_last_infer_timing("unseen-request", &timing));
}

}  // namespace

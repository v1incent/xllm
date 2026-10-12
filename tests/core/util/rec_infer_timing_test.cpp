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

#include "core/util/rec_infer_timing.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace xllm::rec_infer_timing {
namespace {
TEST(RecInferTimingTest, RecordsAcrossThreadsAndConsumesOnlyOnce) {
  begin_request("timing-thread-test");
  std::thread writer([] { set_build_request_us("timing-thread-test", 31); });
  writer.join();
  InferTimingDetail detail;
  ASSERT_TRUE(take("timing-thread-test", &detail));
  EXPECT_EQ(detail.build_request_us, 31);
  EXPECT_FALSE(take("timing-thread-test", &detail));
  set_build_request_us("timing-thread-test", 19);
  EXPECT_FALSE(take("timing-thread-test", &detail));
}
TEST(RecInferTimingTest, EvictsOldestUnconsumedEntry) {
  begin_request("evicted-timing");
  for (int32_t i = 0; i < 4096; ++i) {
    begin_request("capacity-" + std::to_string(i));
  }
  InferTimingDetail detail;
  EXPECT_FALSE(take("evicted-timing", &detail));
  for (int32_t i = 0; i < 4096; ++i) {
    EXPECT_TRUE(take("capacity-" + std::to_string(i), &detail));
  }
}
}  // namespace
}  // namespace xllm::rec_infer_timing

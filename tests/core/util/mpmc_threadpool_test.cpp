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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "core/util/threadpool.h"

namespace xllm {
namespace {

TEST(MPMCThreadPoolTest, DestructionWakesIdleWorkers) {
  MPMCThreadPool pool(/*num_threads=*/16);
  // Let construction finish and workers block with no producer to wake them.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

TEST(MPMCThreadPoolTest, DestructionDrainsAllQueuedTasks) {
  std::atomic<int32_t> completed{0};
  constexpr int32_t kTaskCount = 512;
  {
    MPMCThreadPool pool(/*num_threads=*/4);
    for (int32_t index = 0; index < kTaskCount; ++index) {
      pool.schedule([&completed]() {
        completed.fetch_add(1, std::memory_order_relaxed);
      });
    }
  }
  EXPECT_EQ(completed.load(std::memory_order_relaxed), kTaskCount);
}

}  // namespace
}  // namespace xllm

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

#include <Python.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>

#include <atomic>
#include <thread>
#include <vector>

#include "core/platform/npu/python_runtime.h"

namespace xllm::npu {
namespace {

TEST(NpuPythonRuntimeTest, InitializesBeforeAllocationAndPreservesCallerGil) {
  ASSERT_EQ(Py_IsInitialized(), 0);
  ASSERT_TRUE(init_python_runtime(/*device_index=*/0));
  ASSERT_NE(Py_IsInitialized(), 0);
  EXPECT_EQ(PyGILState_Check(), 0);
  torch_npu::init_npu(static_cast<int8_t>(0));
  const auto tensor = torch::zeros(
      {2, 2},
      torch::TensorOptions().device(torch::Device(torch::kPrivateUse1, 0)));
  EXPECT_FLOAT_EQ(tensor.sum().item<float>(), 0.0F);

  const PyGILState_STATE gil = PyGILState_Ensure();
  EXPECT_TRUE(init_python_runtime(/*device_index=*/0));
  EXPECT_NE(PyGILState_Check(), 0);
  PyGILState_Release(gil);

  std::atomic<int32_t> successes{0};
  std::vector<std::thread> callers;
  callers.reserve(2);
  for (int32_t i = 0; i < 2; ++i) {
    callers.emplace_back([&] {
      if (init_python_runtime(/*device_index=*/0)) {
        ++successes;
      }
    });
  }
  for (auto& caller : callers) {
    caller.join();
  }
  EXPECT_EQ(successes.load(), 2);
}

}  // namespace
}  // namespace xllm::npu

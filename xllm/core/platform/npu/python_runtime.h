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

#include <cstdint>

namespace xllm::npu {

// Initialize Python/torch_npu before the first C++ NPU operation and select a
// logical device. Repeated calls are supported, including callers holding the
// GIL. Python remains alive across handler destruction; an external owner must
// not finalize it while xllm is in use.
bool init_python_runtime(int32_t device_index);

}  // namespace xllm::npu

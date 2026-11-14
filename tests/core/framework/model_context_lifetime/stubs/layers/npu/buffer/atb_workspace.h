/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <acl/acl.h>
#include <torch/torch.h>
namespace atb {
using Status = int32_t;
constexpr Status NO_ERROR = 0;
class Context final {
 public:
  explicit Context(aclrtContext runtime) : runtime_(runtime) {}
  Status SetExecuteStream(void* stream);
  Status SetAsyncTilingCopyStatus(bool enabled);
  aclrtContext runtime_;
};
Status CreateContext(Context** context);
Status DestroyContext(Context* context);
}
namespace xllm {
class AtbWorkspace final {
 public:
  explicit AtbWorkspace(torch::Device /*device*/) {}
};
}

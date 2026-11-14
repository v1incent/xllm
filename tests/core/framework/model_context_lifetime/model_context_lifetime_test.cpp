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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/model_context.h"

namespace {
struct RuntimeToken { int32_t device; };
RuntimeToken runtime_tokens[] = {{0}, {1}};
thread_local aclrtContext current_runtime = nullptr;
std::atomic<int32_t> creates{0};
std::atomic<int32_t> destroys{0};
std::atomic<int32_t> active{0};
std::atomic<bool> fail_stream{false};

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

xllm::ModelContext make_context(int32_t device = 0) {
  return xllm::ModelContext(xllm::ParallelArgs(0, 1, nullptr),
                            xllm::ModelArgs(), xllm::QuantArgs(),
                            torch::TensorOptions(torch::Device(device)));
}

void baseline() {
  { auto owner = make_context(); }
  expect(creates == 1 && destroys == 1 && active == 0,
         "ATB context was not released at model teardown");
}

void copies() {
  auto owner = std::make_unique<xllm::ModelContext>(make_context());
  const atb::Context* pointer = owner->get_atb_context();
  auto copy = std::make_unique<xllm::ModelContext>(*owner);
  auto moved = std::make_unique<xllm::ModelContext>(std::move(*copy));
  expect(copy->get_atb_context() == nullptr, "moved-from owner is not empty");
  owner.reset();
  copy.reset();
  expect(active == 1 && moved->get_atb_context() == pointer,
         "copy did not retain the shared ATB context");
  moved.reset();
  expect(creates == 1 && destroys == 1, "copy/move double-freed or leaked");
}

void derived() {
  auto owner = std::make_unique<xllm::ModelContext>(make_context());
  owner->set_model_id("test-model");
  auto parallel = std::make_unique<xllm::ModelContext>(
      owner->with_parallel_args(xllm::ParallelArgs(1, 2, nullptr)));
  xllm::QuantArgs quant;
  quant.quant_method("test-quant");
  auto quantized = std::make_unique<xllm::ModelContext>(
      parallel->with_quant_args(quant));
  expect(parallel->get_parallel_args().rank() == 1 &&
             quantized->get_quant_args().quant_method() == "test-quant" &&
             quantized->get_model_id() == "test-model",
         "derived model metadata changed");
  expect(owner->get_atb_context() == quantized->get_atb_context(),
         "derived context changed the ATB instance");
  owner.reset();
  parallel.reset();
  expect(active == 1, "derived context was freed early");
  quantized.reset();
  expect(creates == 1 && destroys == 1, "derived context leaked");
}

void assignment() {
  auto first = make_context();
  auto second = make_context(1);
  const atb::Context* pointer = first.get_atb_context();
  second = first;
  expect(destroys == 1 && second.get_atb_context() == pointer,
         "copy assignment leaked its previous resource");
  first = make_context(1);
  expect(active == 2, "temporary assignment released a live context");
  second = std::move(first);
  expect(destroys == 2 && first.get_atb_context() == nullptr,
         "move assignment leaked or retained a stale handle");
}

void borrowed() {
  atb::Context* external = nullptr;
  aclrtSetDevice(0);
  atb::CreateContext(&external);
  {
    xllm::ModelContext borrowed(xllm::ParallelArgs(0, 1, nullptr),
                                xllm::ModelArgs(), xllm::QuantArgs(),
                                torch::TensorOptions(), external);
    auto copy = borrowed.with_parallel_args(xllm::ParallelArgs(0, 2, nullptr));
    auto derived = copy.with_quant_args(xllm::QuantArgs());
    expect(derived.get_atb_context() == external, "borrowed pointer changed");
  }
  expect(destroys == 0, "borrowed ATB context was destroyed by the borrower");
  atb::DestroyContext(external);
}

void empty() {
  xllm::ModelContext context;
  expect(context.get_atb_context() == nullptr, "default context not null");
  auto derived = context.with_quant_args(xllm::QuantArgs());
  expect(derived.get_atb_context() == nullptr, "empty derived context not null");
}

void unwind() {
  fail_stream = true;
  bool caught = false;
  try { auto owner = make_context(); }
  catch (const std::runtime_error&) { caught = true; }
  fail_stream = false;
  expect(caught && creates == 1 && destroys == 1,
         "constructor unwinding leaked its acquired ATB context");
}

void consumer() {
  std::shared_ptr<atb::Context> layer_handle;
  {
    auto owner = make_context();
    layer_handle = owner.get_atb_context_handle();
  }
  expect(active == 1, "model teardown invalidated a remaining layer");
  layer_handle.reset();
  expect(destroys == 1, "last layer did not release the context");
}

void threads() {
  auto owner = std::make_unique<xllm::ModelContext>(make_context());
  std::atomic<int32_t> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> workers;
  workers.reserve(8);
  for (int32_t index = 0; index < 8; ++index) {
    workers.emplace_back([copy = *owner, &ready, &release]() mutable {
      aclrtSetDevice(1);
      ready.fetch_add(1);
      while (!release.load()) { std::this_thread::yield(); }
      copy = xllm::ModelContext();
      expect(current_runtime == &runtime_tokens[1],
             "teardown did not restore the calling thread runtime context");
    });
  }
  while (ready.load() != 8) { std::this_thread::yield(); }
  owner.reset();
  expect(active == 1, "threads did not hold shared ownership");
  release = true;
  for (auto& worker : workers) { worker.join(); }
  expect(destroys == 1, "concurrent final releases leaked or double-freed");
  auto next = std::make_unique<xllm::ModelContext>(make_context());
  std::thread fresh([copy = std::move(next)]() mutable {
    expect(current_runtime == nullptr, "fresh thread unexpectedly bound");
    copy.reset();
  });
  fresh.join();
  expect(destroys == 2, "unbound teardown thread leaked");
}

void reload() {
  for (int32_t round = 0; round < 5; ++round) {
    {
      std::vector<xllm::ModelContext> contexts;
      contexts.reserve(6);
      for (int32_t index = 0; index < 6; ++index) {
        contexts.emplace_back(make_context(index / 3));
      }
      expect(active == 6, "unexpected number of model contexts");
    }
    expect(active == 0, "model reload retains ATB contexts");
  }
  expect(creates == 30 && destroys == 30, "reload create/destroy imbalance");
}
}  // namespace

aclError aclrtSetDevice(int32_t device) {
  expect(device >= 0 && device < 2, "bad test device");
  current_runtime = &runtime_tokens[device];
  return ACL_SUCCESS;
}
aclError aclrtGetCurrentContext(aclrtContext* context) {
  *context = current_runtime;
  return current_runtime == nullptr ? 1 : ACL_SUCCESS;
}
aclError aclrtSetCurrentContext(aclrtContext context) {
  current_runtime = context;
  return ACL_SUCCESS;
}
namespace atb {
Status CreateContext(Context** context) {
  *context = new Context(current_runtime);
  creates.fetch_add(1);
  active.fetch_add(1);
  return NO_ERROR;
}
Status DestroyContext(Context* context) {
  expect(context->runtime_ == current_runtime, "destroyed on wrong runtime context");
  destroys.fetch_add(1);
  active.fetch_sub(1);
  delete context;
  return NO_ERROR;
}
Status Context::SetExecuteStream(void* /*stream*/) {
  if (fail_stream) { throw std::runtime_error("injected stream setup failure"); }
  return NO_ERROR;
}
Status Context::SetAsyncTilingCopyStatus(bool /*enabled*/) { return NO_ERROR; }
}  // namespace atb

int main(int argc, char** argv) {
  expect(argc == 2, "pass a test mode");
  const std::string mode = argv[1];
  if (mode == "baseline") { baseline(); }
  else if (mode == "copies") { copies(); }
  else if (mode == "derived") { derived(); }
  else if (mode == "assignment") { assignment(); }
  else if (mode == "borrowed") { borrowed(); }
  else if (mode == "empty") { empty(); }
  else if (mode == "unwind") { unwind(); }
  else if (mode == "consumer") { consumer(); }
  else if (mode == "threads") { threads(); }
  else if (mode == "reload") { reload(); }
  else { expect(false, "unknown test mode"); }
  expect(active == 0 && creates == destroys, "final resource imbalance");
  std::cout << mode << " passed\n";
  return 0;
}

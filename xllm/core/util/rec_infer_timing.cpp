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

#include <list>
#include <mutex>
#include <unordered_map>

namespace xllm::rec_infer_timing {
namespace {
constexpr size_t kMaxDetails = 4096;
std::mutex details_mutex;
std::list<std::string> request_order;
struct Entry {
  InferTimingDetail detail;
  std::list<std::string>::iterator order;
};
std::unordered_map<std::string, Entry> details;

void initialize_detail(const std::string& request_id, bool reset) {
  if (request_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    if (reset) {
      it->second.detail = {};
    }
    return;
  }
  if (details.size() == kMaxDetails) {
    details.erase(request_order.front());
    request_order.pop_front();
  }
  request_order.emplace_back(request_id);
  details.emplace(request_id, Entry{{}, std::prev(request_order.end())});
}
}  // namespace

void begin_request(const std::string& request_id) {
  initialize_detail(request_id, true);
}
void ensure_request(const std::string& request_id) {
  initialize_detail(request_id, false);
}

void set_convert_input_tensors_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.convert_input_tensors_us = us;
  }
}

void set_threadpool_wait_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.threadpool_wait_us = us;
  }
}

void set_build_request_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.build_request_us = us;
  }
}

void set_scheduler_infer_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.scheduler_infer_us = us;
  }
}

void set_generate_output_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.generate_output_us = us;
  }
}

void set_build_response_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(details_mutex);
  auto it = details.find(request_id);
  if (it != details.end()) {
    it->second.detail.build_response_us = us;
  }
}

bool take(const std::string& request_id, InferTimingDetail* out) {
  if (out == nullptr || request_id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(details_mutex);
  const auto it = details.find(request_id);
  if (it == details.end()) {
    return false;
  }
  *out = it->second.detail;
  request_order.erase(it->second.order);
  details.erase(it);
  return true;
}
}  // namespace xllm::rec_infer_timing

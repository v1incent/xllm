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
#include <cstdlib>
#include <iostream>
namespace test {
class Log final {
 public:
  explicit Log(bool fail) : fail_(fail) {}
  ~Log() { if (fail_) { std::abort(); } }
  template <typename Value> Log& operator<<(const Value& value) {
    if (fail_) { std::cerr << value; }
    return *this;
  }
 private:
  bool fail_;
};
}
#define CHECK(condition) test::Log(!(condition))
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define LOG(level) std::cerr

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

#include "core/framework/request/rec_item_utils.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {
TEST(RecItemUtilsTest, DeduplicatesAndKeepsFirstMetadataIndex) {
  const std::vector<int64_t> ids{50, 20, 50, 30, 20};
  EXPECT_EQ(select_unique_rec_item_indices(ids, 0, 7),
            (std::vector<size_t>{0, 1, 3}));
  EXPECT_EQ(select_unique_rec_item_indices(ids, 3, 7),
            (std::vector<size_t>{0, 1, 3}));
  EXPECT_TRUE(select_unique_rec_item_indices({}, 3, 7).empty());
}

TEST(RecItemUtilsTest, SamplesUniqueItemsDeterministicallyWithinLimit) {
  std::vector<int64_t> ids(1000);
  std::iota(ids.begin(), ids.end(), 100);
  const std::vector<int64_t> duplicates(ids.begin(), ids.begin() + 100);
  ids.insert(ids.end(), duplicates.begin(), duplicates.end());
  const auto first = select_unique_rec_item_indices(ids, 17, 42);
  EXPECT_EQ(first.size(), 17);
  EXPECT_EQ(first, select_unique_rec_item_indices(ids, 17, 42));
  std::unordered_set<int64_t> selected;
  for (size_t index : first) {
    ASSERT_LT(index, 1000);
    EXPECT_TRUE(selected.insert(ids[index]).second);
  }
}
}  // namespace
}  // namespace xllm

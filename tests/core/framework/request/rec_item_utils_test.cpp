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

#include <numeric>
#include <random>

namespace xllm {
namespace {

class CopyTrackedItem final {
 public:
  explicit CopyTrackedItem(int64_t id) : id_(id) {}
  CopyTrackedItem(const CopyTrackedItem& other) : id_(other.id_) {
    ++copy_count_;
  }
  CopyTrackedItem(CopyTrackedItem&&) noexcept = default;

  int64_t id() const { return id_; }

  inline static size_t copy_count_ = 0;

 private:
  int64_t id_;
};

TEST(RecItemUtilsTest, WithoutSamplingKeepsFirstMetadataIndex) {
  const std::vector<int64_t> ids{50, 20, 50, 30, 20};
  for (int32_t limit : {-1, 0, 5, 10}) {
    EXPECT_EQ(select_unique_rec_item_indices(ids, limit, /*random_seed=*/7),
              (std::vector<size_t>{0, 1, 3}));
  }
  EXPECT_TRUE(select_unique_rec_item_indices({}, /*limit=*/3, /*random_seed=*/7)
                  .empty());
}

TEST(RecItemUtilsTest, SamplesDistinctEntriesDeterministicallyWithinLimit) {
  std::vector<int64_t> ids(1000);
  std::iota(ids.begin(), ids.end(), 100);
  const auto first =
      select_unique_rec_item_indices(ids, /*limit=*/17, /*random_seed=*/42);
  EXPECT_EQ(first.size(), 17);
  EXPECT_EQ(
      first,
      select_unique_rec_item_indices(ids, /*limit=*/17, /*random_seed=*/42));
  std::unordered_set<int64_t> selected;
  for (size_t index : first) {
    ASSERT_LT(index, ids.size());
    EXPECT_TRUE(selected.insert(ids[index]).second);
  }
}

TEST(RecItemUtilsTest, SinglePickSamplesRawEntriesIncludingDuplicateMetadata) {
  // Index 1 represents different metadata for the same item as index 0.
  const std::vector<int64_t> ids{50, 50, 20};
  bool selected_later_duplicate = false;
  for (int32_t seed = 0; seed < 32; ++seed) {
    std::mt19937 generator(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<size_t> distribution(0, ids.size() - 1);
    const size_t expected = distribution(generator);
    const auto indices = select_unique_rec_item_indices(ids, /*limit=*/1, seed);
    ASSERT_EQ(indices.size(), 1);
    EXPECT_EQ(indices.front(), expected);
    selected_later_duplicate |= indices.front() == 1;
  }
  EXPECT_TRUE(selected_later_duplicate);
}

TEST(RecItemUtilsTest, DeduplicatesAfterSamplingWithoutRefilling) {
  const std::vector<int64_t> ids{50, 50, 50, 50, 20, 30};
  bool returned_fewer_than_limit = false;
  for (int32_t seed = 0; seed < 32; ++seed) {
    const auto indices = select_unique_rec_item_indices(ids, /*limit=*/3, seed);
    ASSERT_LE(indices.size(), 3);
    std::unordered_set<int64_t> selected;
    for (size_t index : indices) {
      ASSERT_LT(index, ids.size());
      EXPECT_TRUE(selected.insert(ids[index]).second);
    }
    returned_fewer_than_limit |= indices.size() < 3;
  }
  EXPECT_TRUE(returned_fewer_than_limit);
}

TEST(RecItemUtilsTest, FixedSeedIncludesSequenceIndex) {
  std::vector<int64_t> ids(1000);
  std::iota(ids.begin(), ids.end(), 100);
  const auto indices = select_unique_rec_item_indices(
      ids, /*limit=*/17, /*random_seed=*/42, /*sequence_index=*/7);
  EXPECT_EQ(
      indices,
      select_unique_rec_item_indices(ids, /*limit=*/17, /*random_seed=*/49));
  EXPECT_NE(
      indices,
      select_unique_rec_item_indices(ids, /*limit=*/17, /*random_seed=*/42));
}

TEST(RecItemUtilsTest, AppendsAllEntriesForUncappedOrLargeLimit) {
  const std::vector<int64_t> ids{50, 50, 20};
  for (int32_t limit : {-1, 0, 3, 10}) {
    std::vector<int64_t> out{99};
    append_sampled_rec_items(
        ids, limit, /*random_seed=*/7, /*sequence_index=*/0, &out);
    EXPECT_EQ(out, (std::vector<int64_t>{99, 50, 50, 20}));
  }
  std::vector<int64_t> out{99};
  append_sampled_rec_items(std::vector<int64_t>{},
                           /*limit=*/1,
                           /*random_seed=*/7,
                           /*sequence_index=*/0,
                           &out);
  EXPECT_EQ(out, (std::vector<int64_t>{99}));
  append_sampled_rec_items<int64_t>(ids,
                                    /*limit=*/1,
                                    /*random_seed=*/7,
                                    /*sequence_index=*/0,
                                    nullptr);
}

TEST(RecItemUtilsTest, LargeListCopiesOnlySelectedEntries) {
  constexpr size_t kItemCount = 100000;
  std::vector<CopyTrackedItem> items;
  items.reserve(kItemCount);
  for (size_t index = 0; index < kItemCount; ++index) {
    items.emplace_back(static_cast<int64_t>(index));
  }
  for (int32_t limit : {1, 17}) {
    std::vector<CopyTrackedItem> out;
    out.reserve(static_cast<size_t>(limit) + 1);
    out.emplace_back(/*id=*/-1);
    CopyTrackedItem::copy_count_ = 0;
    append_sampled_rec_items(
        items, limit, /*random_seed=*/42, /*sequence_index=*/0, &out);
    ASSERT_EQ(out.size(), static_cast<size_t>(limit) + 1);
    EXPECT_EQ(out.front().id(), -1);
    EXPECT_EQ(CopyTrackedItem::copy_count_, static_cast<size_t>(limit));
    std::unordered_set<int64_t> selected;
    for (size_t index = 1; index < out.size(); ++index) {
      const int64_t id = out[index].id();
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), items.size());
      EXPECT_EQ(items[static_cast<size_t>(id)].id(), id);
      EXPECT_TRUE(selected.insert(id).second);
    }
    EXPECT_EQ(items.size(), kItemCount);
  }
}

TEST(RecItemUtilsTest, UnseededSamplingReturnsDistinctEntriesWithinLimit) {
  std::vector<int64_t> ids(1000);
  std::iota(ids.begin(), ids.end(), 100);
  const auto indices =
      select_unique_rec_item_indices(ids, /*limit=*/17, /*random_seed=*/-1);
  ASSERT_EQ(indices.size(), 17);
  const std::unordered_set<size_t> selected(indices.begin(), indices.end());
  EXPECT_EQ(selected.size(), indices.size());
  for (size_t index : indices) {
    EXPECT_LT(index, ids.size());
  }
}
}  // namespace
}  // namespace xllm

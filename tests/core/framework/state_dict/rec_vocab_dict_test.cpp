/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/state_dict/rec_vocab_dict.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"

namespace xllm {
namespace {

class ScopedConstrainedDecodingFlag final {
 public:
  explicit ScopedConstrainedDecodingFlag(bool value)
      : old_value_(RecConfig::get_instance().enable_constrained_decoding()) {
    RecConfig::get_instance().enable_constrained_decoding(value);
  }

  ~ScopedConstrainedDecodingFlag() {
    RecConfig::get_instance().enable_constrained_decoding(old_value_);
  }

 private:
  bool old_value_;
};

class RecVocabDictSamplingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RecConfig::get_instance().each_conversion_threshold(1);
    RecConfig::get_instance().enable_extended_item_info(false);
    ExecutionConfig::get_instance().random_seed(42);
    vocab_path_ =
        std::filesystem::path(::testing::TempDir()) /
        (std::string(
             ::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".bin");
  }

  void TearDown() override {
    std::filesystem::remove(vocab_path_);
    RecConfig::get_instance().each_conversion_threshold(old_limit_);
    RecConfig::get_instance().enable_extended_item_info(old_extended_);
    ExecutionConfig::get_instance().random_seed(old_seed_);
  }

  std::filesystem::path vocab_path_;

 private:
  const int32_t old_limit_ =
      RecConfig::get_instance().each_conversion_threshold();
  const bool old_extended_ =
      RecConfig::get_instance().enable_extended_item_info();
  const int32_t old_seed_ = ExecutionConfig::get_instance().random_seed();
};

void write_vocab_file(
    const std::filesystem::path& path,
    const std::vector<std::pair<int64_t, RecTokenTriple>>& records) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  CHECK(ofs.is_open()) << "Failed to open test vocab file: " << path;
  for (const auto& record : records) {
    const int64_t item_id = record.first;
    const RecTokenTriple& tokens = record.second;
    ofs.write(reinterpret_cast<const char*>(&item_id), sizeof(item_id));
    ofs.write(reinterpret_cast<const char*>(tokens.data()),
              REC_TOKEN_SIZE * sizeof(int32_t));
  }
}

void write_extended_vocab_file(const std::filesystem::path& path,
                               const std::vector<RecItemInfo>& infos,
                               const RecTokenTriple& tokens) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  CHECK(ofs.is_open()) << "Failed to open test vocab file: " << path;
  for (const auto& info : infos) {
    ofs.write(reinterpret_cast<const char*>(&info.item_id),
              sizeof(info.item_id));
    for (const auto& field : {info.did, info.type}) {
      const uint32_t length = static_cast<uint32_t>(field.size());
      ofs.write(reinterpret_cast<const char*>(&length), sizeof(length));
      ofs.write(field.data(), length);
    }
    ofs.write(reinterpret_cast<const char*>(tokens.data()),
              REC_TOKEN_SIZE * sizeof(int32_t));
  }
}

std::vector<int32_t> sorted_next_tokens(
    const std::unordered_set<int32_t>& token_set) {
  std::vector<int32_t> tokens(token_set.begin(), token_set.end());
  std::sort(tokens.begin(), tokens.end());
  return tokens;
}

std::vector<int32_t> prefix1_values_for_token(const RecConstraintTables& tables,
                                              int32_t t0) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  return std::vector<int32_t>(tables.prefix1_values.begin() + begin,
                              tables.prefix1_values.begin() + end);
}

std::vector<int32_t> prefix2_values_for_tokens(
    const RecConstraintTables& tables,
    int32_t t0,
    int32_t t1) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  for (int32_t index = begin; index < end; ++index) {
    if (tables.prefix1_values[static_cast<size_t>(index)] != t1) {
      continue;
    }
    const int32_t prefix2_begin =
        tables.prefix2_value_offsets[static_cast<size_t>(index)];
    const int32_t prefix2_end =
        tables.prefix2_value_offsets[static_cast<size_t>(index) + 1];
    return std::vector<int32_t>(tables.prefix2_values.begin() + prefix2_begin,
                                tables.prefix2_values.begin() + prefix2_end);
  }
  return {};
}

std::vector<int64_t> prefix1_pair_keys_for_token(
    const RecConstraintTables& tables,
    int32_t t0) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  return std::vector<int64_t>(tables.prefix1_pair_keys.begin() + begin,
                              tables.prefix1_pair_keys.begin() + end);
}

}  // namespace

TEST_F(RecVocabDictSamplingTest, CapsLargeLookupBeforeReturningIdsOrMetadata) {
  constexpr size_t kItemCount = 10000;
  const RecTokenTriple tokens{1, 2, 3};
  std::vector<std::pair<int64_t, RecTokenTriple>> records;
  records.reserve(kItemCount);
  for (size_t index = 0; index < kItemCount; ++index) {
    records.emplace_back(static_cast<int64_t>(index) + 100, tokens);
  }
  write_vocab_file(vocab_path_, records);
  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path_.string()));

  // Dictionary sampling uses sequence index zero, even when appending.
  std::mt19937 generator(/*seed=*/42);
  std::uniform_int_distribution<size_t> distribution(0, kItemCount - 1);
  const int64_t expected_id = records[distribution(generator)].first;
  std::vector<int64_t> ids{99};
  ASSERT_TRUE(vocab_dict.get_items_by_tokens(tokens, &ids));
  EXPECT_EQ(ids, (std::vector<int64_t>{99, expected_id}));
  std::vector<RecItemInfo> infos;
  ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(tokens, &infos));
  ASSERT_EQ(infos.size(), 1);
  EXPECT_EQ(infos.front().item_id, expected_id);
  EXPECT_TRUE(infos.front().did.empty());
  EXPECT_TRUE(infos.front().type.empty());

  RecConfig::get_instance().each_conversion_threshold(17);
  ids.clear();
  infos.clear();
  ASSERT_TRUE(vocab_dict.get_items_by_tokens(tokens, &ids));
  ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(tokens, &infos));
  ASSERT_EQ(ids.size(), 17);
  ASSERT_EQ(infos.size(), ids.size());
  EXPECT_EQ(std::unordered_set<int64_t>(ids.begin(), ids.end()).size(), 17);
  for (size_t index = 0; index < ids.size(); ++index) {
    EXPECT_EQ(ids[index], infos[index].item_id);
    EXPECT_GE(ids[index], 100);
    EXPECT_LT(ids[index], static_cast<int64_t>(kItemCount) + 100);
  }

  // Sampling the forward lookup must not prune the dictionary itself.
  std::vector<int32_t> encoded;
  ASSERT_TRUE(vocab_dict.get_tokens_by_item(records.back().first, &encoded));
  EXPECT_EQ(encoded, (std::vector<int32_t>{1, 2, 3}));
}

TEST_F(RecVocabDictSamplingTest,
       UncappedLookupPreservesDuplicatesAndAppendOrder) {
  const RecTokenTriple tokens{1, 2, 3};
  write_vocab_file(vocab_path_, {{50, tokens}, {50, tokens}, {20, tokens}});
  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path_.string()));
  for (int32_t limit : {-1, 0, 3, 10}) {
    RecConfig::get_instance().each_conversion_threshold(limit);
    std::vector<int64_t> ids{99};
    ASSERT_TRUE(vocab_dict.get_items_by_tokens(tokens, &ids));
    EXPECT_EQ(ids, (std::vector<int64_t>{99, 50, 50, 20}));
  }
  std::vector<int64_t> ids{99};
  EXPECT_FALSE(vocab_dict.get_items_by_tokens(RecTokenTriple{7, 8, 9}, &ids));
  EXPECT_EQ(ids, (std::vector<int64_t>{99}));
  std::vector<RecItemInfo> infos{{99, "original", "original"}};
  EXPECT_FALSE(
      vocab_dict.get_item_infos_by_tokens(RecTokenTriple{7, 8, 9}, &infos));
  ASSERT_EQ(infos.size(), 1);
  EXPECT_EQ(infos.front().item_id, 99);
  EXPECT_EQ(infos.front().did, "original");
}

TEST_F(RecVocabDictSamplingTest, SamplesMatchingExtendedMetadata) {
  RecConfig::get_instance().enable_extended_item_info(true);
  const RecTokenTriple tokens{1, 2, 3};
  // Different entries can carry distinct metadata for the same item ID.
  const std::vector<RecItemInfo> records{{50, "did-first", "type-first"},
                                         {50, "did-second", "type-second"},
                                         {20, "did-third", "type-third"}};
  write_extended_vocab_file(vocab_path_, records, tokens);
  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path_.string()));
  bool selected_later_duplicate = false;
  for (int32_t seed = 0; seed < 32; ++seed) {
    ExecutionConfig::get_instance().random_seed(seed);
    std::mt19937 generator(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<size_t> distribution(0, records.size() - 1);
    const size_t expected_index = distribution(generator);
    const auto& expected = records[expected_index];
    std::vector<RecItemInfo> infos{{99, "original", "original"}};
    ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(tokens, &infos));
    ASSERT_EQ(infos.size(), 2);
    EXPECT_EQ(infos.front().item_id, 99);
    EXPECT_EQ(infos.front().did, "original");
    EXPECT_EQ(infos.back().item_id, expected.item_id);
    EXPECT_EQ(infos.back().did, expected.did);
    EXPECT_EQ(infos.back().type, expected.type);
    std::vector<int64_t> ids;
    ASSERT_TRUE(vocab_dict.get_items_by_tokens(tokens, &ids));
    EXPECT_EQ(ids, (std::vector<int64_t>{expected.item_id}));
    selected_later_duplicate |= expected_index == 1;
  }
  EXPECT_TRUE(selected_later_duplicate);
}

TEST(RecVocabDictTest, BuildConstraintTablesMatchesLegacyPrefixMap) {
  ScopedConstrainedDecodingFlag flag(/*value=*/true);
  const std::filesystem::path vocab_path =
      std::filesystem::path(::testing::TempDir()) / "rec_vocab_dict_test.bin";

  write_vocab_file(vocab_path,
                   {
                       {100, RecTokenTriple{1, 2, 3}},
                       {101, RecTokenTriple{1, 2, 4}},
                       {102, RecTokenTriple{1, 5, 6}},
                       {103, RecTokenTriple{7, 8, 9}},
                       {104, RecTokenTriple{7, 8, 10}},
                       {105, RecTokenTriple{1, 2, 3}},
                   });

  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path.string()));

  const RecConstraintTables tables =
      vocab_dict.build_constraint_tables(/*vocab_size=*/16);

  EXPECT_EQ(tables.vocab_size, 16);
  EXPECT_EQ(tables.first_token_ids, std::vector<int32_t>({1, 7}));
  EXPECT_EQ(prefix1_values_for_token(tables, /*t0=*/1),
            std::vector<int32_t>({2, 5}));
  EXPECT_EQ(prefix1_values_for_token(tables, /*t0=*/7),
            std::vector<int32_t>({8}));
  EXPECT_EQ(prefix1_pair_keys_for_token(tables, /*t0=*/1),
            std::vector<int64_t>({18, 21}));
  EXPECT_EQ(prefix1_pair_keys_for_token(tables, /*t0=*/7),
            std::vector<int64_t>({120}));
  EXPECT_TRUE(std::is_sorted(tables.prefix1_pair_keys.begin(),
                             tables.prefix1_pair_keys.end()));
  EXPECT_EQ(tables.prefix1_pair_keys.size(), tables.prefix1_values.size());
  EXPECT_TRUE(prefix1_values_for_token(tables, /*t0=*/0).empty());
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/2),
            std::vector<int32_t>({3, 4}));
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/5),
            std::vector<int32_t>({6}));
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/7, /*t1=*/8),
            std::vector<int32_t>({9, 10}));
  EXPECT_EQ(tables.prefix2_value_offsets.size(),
            tables.prefix1_values.size() + 1);
  EXPECT_EQ(tables.max_first_degree, 2);
  EXPECT_EQ(tables.max_prefix1_degree, 2);
  EXPECT_EQ(tables.max_prefix2_degree, 2);

  std::vector<int32_t> empty_prefix;
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(empty_prefix))),
            tables.first_token_ids);

  std::vector<int32_t> prefix1{1};
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(prefix1))),
            prefix1_values_for_token(tables, /*t0=*/1));

  std::vector<int32_t> prefix2{1, 2};
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(prefix2))),
            prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/2));

  std::filesystem::remove(vocab_path);
}

}  // namespace xllm

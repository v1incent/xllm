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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/common/version_singleton.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/request/request_output.h"
#include "core/framework/state_dict/rec_vocab_dict.h"
#include "core/framework/tokenizer/rec_tokenizer.h"
#include "core/framework/tokenizer/tokenizer_proxy.h"

namespace xllm {
namespace {

class IdsOnlyTokenizer final : public Tokenizer {
 public:
  bool decode(const Slice<int32_t>& /*token_ids*/,
              bool /*skip_special_tokens*/,
              std::vector<int64_t>* item_ids) const override {
    item_ids->emplace_back(101);
    return true;
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<IdsOnlyTokenizer>();
  }
};

// Each worker owns a fresh thread-local tokenizer, as in the serving path.
SequenceOutput decode_with_proxy(std::unique_ptr<Tokenizer> tokenizer,
                                 std::vector<int32_t> token_ids,
                                 bool clone_proxy = false) {
  SequenceOutput output{};
  output.token_ids = std::move(token_ids);
  std::thread worker([&] {
    TokenizerProxy proxy(std::move(tokenizer));
    std::unique_ptr<Tokenizer> cloned;
    const Tokenizer* decoder = &proxy;
    if (clone_proxy) {
      cloned = proxy.clone();
      decoder = cloned.get();
    }
    decode_rec_items(*decoder, /*skip_special_tokens=*/true,
                     /*sequence_index=*/0, output);
  });
  worker.join();
  return output;
}

class RecItemDecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_config_ = RecConfig::get_instance();
    saved_seed_ = ExecutionConfig::get_instance().random_seed();
    RecConfig::get_instance()
        .enable_convert_tokens_to_item(true)
        .enable_extended_item_info(true)
        .enable_constrained_decoding(false)
        .each_conversion_threshold(0);
    ExecutionConfig::get_instance().random_seed(42);
    model_version_ =
        "rec_item_decode_" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    vocab_path_ = std::filesystem::path(::testing::TempDir()) /
                  (model_version_ + ".bin");
    const std::vector<std::pair<RecItemInfo, RecTokenTriple>> records{
        {{101, "did-101", "type-1"}, {1, 2, 3}},
        {{202, "did-202", "type-2"}, {4, 5, 6}}};
    std::ofstream stream(vocab_path_, std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    for (const auto& [info, tokens] : records) {
      stream.write(reinterpret_cast<const char*>(&info.item_id),
                   sizeof(info.item_id));
      for (const std::string& field : {info.did, info.type}) {
        const uint32_t length = static_cast<uint32_t>(field.size());
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        stream.write(field.data(), length);
      }
      stream.write(reinterpret_cast<const char*>(tokens.data()),
                   tokens.size() * sizeof(int32_t));
    }
    stream.close();
    ASSERT_TRUE(stream.good());
    ASSERT_TRUE(VersionSingleton<RecVocabDict>::GetInstance(model_version_)
                    ->initialize(vocab_path_.string()));
  }

  void TearDown() override {
    std::filesystem::remove(vocab_path_);
    RecConfig::get_instance() = saved_config_;
    ExecutionConfig::get_instance().random_seed(saved_seed_);
  }

  std::unique_ptr<Tokenizer> make_tokenizer() const {
    return std::make_unique<RecTokenizer>(model_version_, TokenizerArgs{});
  }

  void expect_metadata(const SequenceOutput& output,
                       const std::vector<int64_t>& expected_ids) const {
    ASSERT_EQ(output.item_ids_list, expected_ids);
    ASSERT_EQ(output.item_infos_list.size(), expected_ids.size());
    for (size_t index = 0; index < expected_ids.size(); ++index) {
      const RecItemInfo& info = output.item_infos_list[index];
      EXPECT_EQ(info.item_id, expected_ids[index]);
      EXPECT_EQ(info.did, "did-" + std::to_string(expected_ids[index]));
      EXPECT_EQ(info.type, expected_ids[index] == 101 ? "type-1" : "type-2");
    }
    ASSERT_TRUE(output.item_info.has_value());
    EXPECT_EQ(output.item_ids, expected_ids.front());
    EXPECT_EQ(output.item_info->item_id, expected_ids.front());
    EXPECT_EQ(output.item_info->did, output.item_infos_list.front().did);
    EXPECT_EQ(output.item_info->type, output.item_infos_list.front().type);
  }

 private:
  RecConfig saved_config_;
  int32_t saved_seed_ = -1;
  std::string model_version_;
  std::filesystem::path vocab_path_;
};

TEST_F(RecItemDecodeTest, DirectTokenizerPreservesMetadata) {
  const auto tokenizer = make_tokenizer();
  SequenceOutput output{};
  output.token_ids = {1, 2, 3, 4, 5, 6};
  decode_rec_items(*tokenizer, /*skip_special_tokens=*/true,
                   /*sequence_index=*/0, output);
  expect_metadata(output, {101, 202});
}

TEST_F(RecItemDecodeTest, ProxyPreservesSingleItemMetadata) {
  const auto output = decode_with_proxy(make_tokenizer(), {1, 2, 3});
  expect_metadata(output, {101});
}

TEST_F(RecItemDecodeTest, ProxyPreservesMultipleItemsAndSkipsUnknownTokens) {
  const auto output =
      decode_with_proxy(make_tokenizer(), {1, 2, 3, 7, 8, 9, 4, 5, 6});
  expect_metadata(output, {101, 202});
}

TEST_F(RecItemDecodeTest, ClonedProxyPreservesMetadata) {
  const auto output = decode_with_proxy(make_tokenizer(), {4, 5, 6},
                                        /*clone_proxy=*/true);
  expect_metadata(output, {202});
}

TEST_F(RecItemDecodeTest, DisabledExtendedInfoReturnsOnlyIds) {
  RecConfig::get_instance().enable_extended_item_info(false);
  const auto output = decode_with_proxy(make_tokenizer(), {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(output.item_ids_list, (std::vector<int64_t>{101, 202}));
  EXPECT_EQ(output.item_ids, 101);
  EXPECT_TRUE(output.item_infos_list.empty());
  EXPECT_FALSE(output.item_info.has_value());
}

TEST_F(RecItemDecodeTest, IdsOnlyTokenizerRetainsLegacyFallback) {
  const auto output =
      decode_with_proxy(std::make_unique<IdsOnlyTokenizer>(), {1, 2, 3});
  ASSERT_EQ(output.item_ids_list, (std::vector<int64_t>{101}));
  ASSERT_EQ(output.item_infos_list.size(), 1);
  EXPECT_EQ(output.item_infos_list.front().item_id, 101);
  EXPECT_TRUE(output.item_infos_list.front().did.empty());
  EXPECT_TRUE(output.item_infos_list.front().type.empty());
}

}  // namespace
}  // namespace xllm

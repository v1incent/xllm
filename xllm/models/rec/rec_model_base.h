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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string_view>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#include "core/util/rec_model_utils.h"

namespace xllm {

template <typename ModelType>
class RecForCausalLMImplBase : public torch::nn::Module {
 public:
  explicit RecForCausalLMImplBase(const ModelContext& context)
      : context_(context) {
    const auto& args = context.get_model_args();
    tie_word_embeddings_ = args.tie_word_embeddings();
    const float denom =
        std::sqrt(static_cast<float>(std::max<int64_t>(1, args.hidden_size())));
    scale_factor_ = denom > 0.0f ? (1.0f / denom) : 1.0f;

    model_ = register_module("model", ModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  virtual ModelOutput forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  // OneRec split lm_head: at decode step N, use only lm_head_segments_[N]
  // (shape [seg_width_N, hidden]) to compute that segment's logits.
  //
  // Two output shapes, orthogonal to the single-step vocab-probs feature:
  //  - Multi-round beam search: scatter the segment back into a full
  //    [rows, vocab_size] tensor at seg_offsets_[N] (other positions filled
  //    with a very negative value), preserving the 25000-wide, global-token-id
  //    contract that beam search / constrained topk / embedding all rely on.
  //  - Single-step mode (max_decode_rounds == 1, is_onerec_single_step_mode()):
  //    return the narrow segment [rows, seg_width_N] directly. The sampler's
  //    softmax then yields vocab_probs of the segment width, which the C API
  //    emits as-is (P2 semantics: the caller wants the split-sized probs, not
  //    the padded full vocab). Single-step is always step 0, so the segment
  //    offset is 0 and local index == global token id.
  //
  // Models without split heads retain the full-vocabulary path.
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes,
                               int32_t step) {
    if (!use_split_lm_head_) {
      return logits(hidden_states, selected_idxes);
    }
    CHECK_GE(step, 0);
    CHECK_LT(step, static_cast<int32_t>(lm_head_segments_.size()))
        << "Missing split lm_head for decode round " << step;
    auto h = hidden_states;
    if (tie_word_embeddings_) {
      h = hidden_states * scale_factor_;
    }
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    auto seg = lm_head_segments_[step](h);  // [rows, seg_width_step]
    if (is_onerec_single_step_mode()) {
      return seg;
    }
    const int64_t rows = seg.size(0);
    auto full = torch::full(
        {rows, full_vocab_size_}, kSplitLmHeadNegInf, seg.options());
    full.slice(/*dim=*/1,
               /*start=*/seg_offsets_[step],
               /*end=*/seg_offsets_[step] + seg_widths_[step]) = seg;
    return full;
  }

  virtual torch::Tensor pooler(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  virtual void load_model(std::unique_ptr<ModelLoader> loader,
                          std::string prefix = "model.") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      if (tie_word_embeddings_) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "shared."));
      } else {
        lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
      }
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  virtual void update_expert_weight(int32_t layer_id) { (void)layer_id; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  // Load segments as shards arrive. Segment indices need not be contiguous
  // within a shard; offsets are finalized after the entire checkpoint loads.
  bool try_build_split_lm_head(const StateDict& state_dict) {
    bool found = false;
    constexpr std::string_view kPrefix = "lm_head_";
    constexpr std::string_view kSuffix = ".weight";
    for (const auto& [name, unused_tensor] : state_dict) {
      const std::string_view key(name);
      if (!key.starts_with(kPrefix) || !key.ends_with(kSuffix)) {
        continue;
      }
      const auto digits = key.substr(
          kPrefix.size(), key.size() - kPrefix.size() - kSuffix.size());
      int32_t index = -1;
      const auto parsed =
          std::from_chars(digits.data(), digits.data() + digits.size(), index);
      CHECK(parsed.ec == std::errc{} &&
            parsed.ptr == digits.data() + digits.size() && index >= 0)
          << "Invalid split lm_head name: " << name;
      CHECK(!pending_lm_heads_.contains(index))
          << "Duplicate split lm_head: " << name;
      auto weight = state_dict.get_tensor(name);
      CHECK_EQ(weight.dim(), 2);
      CHECK_GT(weight.size(0), 0);
      CHECK_EQ(weight.size(1), context_.get_model_args().hidden_size());
      const auto& parallel = context_.get_parallel_args();
      auto head =
          register_module("lm_head_" + std::to_string(index),
                          layer::LmHead(context_.get_model_args().hidden_size(),
                                        weight.size(0),
                                        /*bias=*/false,
                                        /*gather_output=*/true,
                                        QuantArgs{},
                                        parallel.lm_head_group_ != nullptr
                                            ? parallel.lm_head_group_
                                            : parallel.tp_group_,
                                        context_.get_tensor_options()));
      head->load_state_dict(state_dict.get_dict_with_prefix(
          "lm_head_" + std::to_string(index) + "."));
      pending_lm_heads_.emplace(index, std::make_pair(weight.size(0), head));
      found = true;
    }
    return found;
  }

  void finalize_split_lm_head() {
    if (pending_lm_heads_.empty()) {
      return;
    }
    full_vocab_size_ = context_.get_model_args().vocab_size();
    lm_head_segments_.reserve(pending_lm_heads_.size());
    seg_widths_.reserve(pending_lm_heads_.size());
    seg_offsets_.reserve(pending_lm_heads_.size());
    int64_t offset = 0;
    for (const auto& [index, segment] : pending_lm_heads_) {
      CHECK_EQ(index, static_cast<int32_t>(lm_head_segments_.size()))
          << "Split lm_head indices must be contiguous starting at zero";
      seg_offsets_.emplace_back(offset);
      seg_widths_.emplace_back(segment.first);
      lm_head_segments_.emplace_back(segment.second);
      offset += segment.first;
    }
    CHECK_LE(offset, full_vocab_size_);
    use_split_lm_head_ = true;
    pending_lm_heads_.clear();
    LOG(INFO) << "OneRec split lm_head enabled: " << lm_head_segments_.size()
              << " segments, total width=" << offset;
  }

  virtual layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  }

 protected:
  float scale_factor_ = 1.0f;
  bool tie_word_embeddings_ = false;
  ModelContext context_;

  // OneRec split lm_head state. When use_split_lm_head_ is true, per-step heads
  // in lm_head_segments_ replace the single lm_head_ for logits computation.
  // Segment count / widths / offsets are inferred from the checkpoint weights
  // (not hardcoded); offsets are the prefix sum of widths. full_vocab_size_ is
  // the scatter target width (= config vocab_size).
  static constexpr float kSplitLmHeadNegInf =
      -std::numeric_limits<float>::infinity();
  bool use_split_lm_head_ = false;
  int64_t full_vocab_size_ = 0;
  std::map<int32_t, std::pair<int64_t, layer::LmHead>> pending_lm_heads_;
  std::vector<layer::LmHead> lm_head_segments_;
  std::vector<int64_t> seg_widths_;
  std::vector<int64_t> seg_offsets_;

  ModelType model_{nullptr};
  layer::LmHead lm_head_{nullptr};
};

}  // namespace xllm

#include "Qwen3ReasoningUtils.hpp"

#include <string>
#include <vector>

#include <llama.h>

#include "utils/LoggingMacros.hpp"

using namespace qvac_lib_inference_addon_cpp::logger;

namespace qvac_lib_inference_addon_llama {
namespace utils {

namespace {

// Qwen3-family default tags. Defined here as the canonical Qwen3 marker
// pair used by `initializeQwen3ReasoningState` so the legacy entry point
// stays byte-for-byte equivalent to the pre-abstraction behaviour.
inline constexpr ReasoningTags kQwen3DefaultTags{
    .open = "<think>",
    .close = "</think>",
};

} // namespace

void initializeReasoningState(
    ::llama_context* lctx, ReasoningState& state, ReasoningTags tags) {
  state.tags = tags;
  state.openTokenCount = 0;
  state.forcedOpenTokenCount = 0;
  state.cached_close_tag_token = LLAMA_TOKEN_NULL;
  state.cached_newline_token = LLAMA_TOKEN_NULL;

  if (lctx == nullptr || tags.open.empty() || tags.close.empty()) {
    return;
  }

  // Tokenise the open marker so the start-position capture path knows
  // how many KV positions the marker occupies. `parse_special=true` is
  // required: Qwen3's `<think>` and Gemma 4's `<|channel>` are special
  // tokens and would otherwise be split into raw pieces.
  std::vector<llama_token> openTokens =
      common_tokenize(lctx, std::string(tags.open), false, true);
  state.openTokenCount = static_cast<int>(openTokens.size());

  // Tokenise the *template-forced* open prefix (`open + "\n"`). Some
  // chat templates (Qwen3 / DeepSeek-R1) end the assistant prefix with
  // `<think>\n` so the model resumes generation inside the reasoning
  // block. Callers that need to compact the cache region covered by
  // that prefix use this count.
  std::vector<llama_token> forcedOpenTokens =
      common_tokenize(lctx, std::string(tags.open) + "\n", false, true);
  state.forcedOpenTokenCount = static_cast<int>(forcedOpenTokens.size());

  // Cache the single-token close-marker id when applicable so the
  // EOS-inside-reasoning replacement path can swap it in without
  // re-tokenising. Multi-token close markers fall back to the
  // substring-detect path (the model closes its own channel).
  std::vector<llama_token> closeTokens =
      common_tokenize(lctx, std::string(tags.close), false, true);
  if (closeTokens.size() == 1) {
    state.cached_close_tag_token = closeTokens[0];
  }

  std::vector<llama_token> newlineTokens =
      common_tokenize(lctx, "\n", false, true);
  if (!newlineTokens.empty()) {
    state.cached_newline_token = newlineTokens[0];
  }
}

void initializeQwen3ReasoningState(
    ::llama_context* lctx, ReasoningState& state) {
  initializeReasoningState(lctx, state, kQwen3DefaultTags);
}

void updateReasoningBuffer(
    const std::string& tokenStr, ReasoningState& state) {
  if (tokenStr.empty()) {
    return;
  }
  state.recent_output_buffer += tokenStr;
  if (state.recent_output_buffer.length() > ReasoningState::BUFFER_SIZE) {
    state.recent_output_buffer = state.recent_output_buffer.substr(
        state.recent_output_buffer.length() - ReasoningState::BUFFER_SIZE);
  }

  if (state.tags.open.empty() || state.tags.close.empty()) {
    return;
  }

  if (state.recent_output_buffer.find(state.tags.open) != std::string::npos) {
    state.inside_reasoning = true;
  }
  if (state.recent_output_buffer.find(state.tags.close) != std::string::npos) {
    state.inside_reasoning = false;
  }
}

void updateQwen3ReasoningBuffer(
    const std::string& tokenStr, ReasoningState& state) {
  updateReasoningBuffer(tokenStr, state);
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

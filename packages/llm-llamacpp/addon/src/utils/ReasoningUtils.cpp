#include "ReasoningUtils.hpp"

#include <string>
#include <vector>

#include <llama.h>

namespace qvac_lib_inference_addon_llama {
namespace utils {

namespace {

// Returns true iff every piece in `tokens` has a CONTROL or USER_DEFINED
// attribute — those are BPE-merge barriers under `parse_special=true`, so
// the standalone tokenisation of a marker matches its in-context emission
// piece-for-piece. Empty `tokens` returns false.
bool allTokensAreSpecial(
    const ::llama_vocab* vocab, const std::vector<llama_token>& tokens) {
  if (tokens.empty() || vocab == nullptr) {
    return false;
  }
  constexpr int specialMask =
      LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED;
  for (const llama_token tok : tokens) {
    const llama_token_attr attr = llama_vocab_get_attr(vocab, tok);
    if ((static_cast<int>(attr) & specialMask) == 0) {
      return false;
    }
  }
  return true;
}

} // namespace

bool initializeReasoningState(
    ::llama_context* lctx, ReasoningState& state, ReasoningTags tags) {
  state.tags = tags;
  state.openTokenCount = 0;
  state.forcedOpenTokenCount = 0;
  state.cached_close_tag_token = LLAMA_TOKEN_NULL;
  state.cached_newline_token = LLAMA_TOKEN_NULL;

  if (lctx == nullptr || tags.open.empty() || tags.close.empty()) {
    return false;
  }

  // Standalone counts match the in-context emission iff every piece of the
  // open marker is a registered special token (CONTROL / USER_DEFINED acts
  // as a BPE-merge barrier under parse_special=true). The span start math
  // `nPast_ - (openTokenCount - 1)` in TextLlmContext relies on this — if a
  // piece is normal text and BPE merges across the boundary at runtime, the
  // in-context emission count would diverge from the standalone count and
  // the recorded span would drop the wrong KV range.
  std::vector<llama_token> openTokens =
      common_tokenize(lctx, tags.open, false, true);
  const ::llama_vocab* vocab = llama_model_get_vocab(llama_get_model(lctx));
  if (!allTokensAreSpecial(vocab, openTokens)) {
    state.tags = ReasoningTags{};
    return false;
  }
  state.openTokenCount = static_cast<int>(openTokens.size());

  std::vector<llama_token> forcedOpenTokens =
      common_tokenize(lctx, tags.open + "\n", false, true);
  state.forcedOpenTokenCount = static_cast<int>(forcedOpenTokens.size());

  std::vector<llama_token> closeTokens =
      common_tokenize(lctx, tags.close, false, true);
  if (closeTokens.size() == 1) {
    state.cached_close_tag_token = closeTokens[0];
  }

  std::vector<llama_token> newlineTokens =
      common_tokenize(lctx, "\n", false, true);
  if (!newlineTokens.empty()) {
    state.cached_newline_token = newlineTokens[0];
  }
  return true;
}

void updateReasoningBuffer(const std::string& tokenStr, ReasoningState& state) {
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

  // Single-block policy in `TextLlmContext::setOpenThinkSpan`: only the
  // first `<think>...</think>` per inference is tracked. A simple
  // independent `find` for each marker is sufficient — the second-block
  // edge case (stale close in buffer when a new open arrives) would
  // matter only if we acted on a second open, which we don't.
  if (state.recent_output_buffer.find(state.tags.open) != std::string::npos) {
    state.inside_reasoning = true;
  }
  if (state.recent_output_buffer.find(state.tags.close) != std::string::npos) {
    state.inside_reasoning = false;
  }
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

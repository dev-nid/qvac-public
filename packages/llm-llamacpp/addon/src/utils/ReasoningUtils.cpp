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

  // The buffer is a rolling window, so a previous block's close marker can
  // still linger when the next block's open marker arrives. A naive
  // independent `find(open)` / `find(close)` would set inside_reasoning=true
  // and then immediately back to false, suppressing the false→true edge and
  // causing the second block to be missed entirely.
  //
  // Use the LATEST occurrence of each marker via `rfind` and let the larger
  // position win — that reflects "the most recent transition wins", which
  // is the right state for the detector consumers (edge-triggered open /
  // close span capture in TextLlmContext::onLogitsReady).
  const size_t lastOpen = state.recent_output_buffer.rfind(state.tags.open);
  const size_t lastClose = state.recent_output_buffer.rfind(state.tags.close);

  if (lastOpen == std::string::npos && lastClose == std::string::npos) {
    return;
  }
  if (lastClose == std::string::npos) {
    state.inside_reasoning = true;
    return;
  }
  if (lastOpen == std::string::npos) {
    state.inside_reasoning = false;
    return;
  }
  state.inside_reasoning = lastOpen > lastClose;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

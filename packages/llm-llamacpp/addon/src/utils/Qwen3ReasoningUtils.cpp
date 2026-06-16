#include "Qwen3ReasoningUtils.hpp"

#include <string>
#include <vector>

#include <llama.h>

namespace qvac_lib_inference_addon_llama {
namespace utils {

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

  std::vector<llama_token> openTokens =
      common_tokenize(lctx, std::string(tags.open), false, true);
  state.openTokenCount = static_cast<int>(openTokens.size());

  std::vector<llama_token> forcedOpenTokens =
      common_tokenize(lctx, std::string(tags.open) + "\n", false, true);
  state.forcedOpenTokenCount = static_cast<int>(forcedOpenTokens.size());

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

  if (state.recent_output_buffer.find(state.tags.open) != std::string::npos) {
    state.inside_reasoning = true;
  }
  if (state.recent_output_buffer.find(state.tags.close) != std::string::npos) {
    state.inside_reasoning = false;
  }
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

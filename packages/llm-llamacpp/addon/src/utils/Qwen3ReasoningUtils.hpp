#pragma once

#include <string>
#include <string_view>

#include "common/common.h"

// Forward declarations from llama.h
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Open / close substring markers used to detect a model's reasoning
// channel in the streamed output (Qwen3 `<think>`/`</think>`,
// Gemma 4 `<|channel>thought`/`<channel|>`, ...).
struct ReasoningTags {
  std::string_view open;
  std::string_view close;
};

struct ReasoningState {
  ReasoningTags tags;
  // Number of tokens the open marker tokenises to under the active
  // tokenizer. Cached at init for span start-position arithmetic.
  int openTokenCount = 0;
  // Token count for `tags.open + "\n"`, the template-forced reasoning
  // prefix some chat templates append to the assistant turn. 0 when
  // not applicable.
  int forcedOpenTokenCount = 0;
  // Cached close-marker id when the marker tokenises to a single
  // token (enables EOS-inside-reasoning replacement).
  llama_token cached_close_tag_token = LLAMA_TOKEN_NULL;
  llama_token cached_newline_token = LLAMA_TOKEN_NULL;
  bool inside_reasoning = false;
  std::string recent_output_buffer;

  static constexpr size_t BUFFER_SIZE = 50;
};

// Initialise `state` with `tags`. Tokenises both markers under
// `lctx`'s vocab to populate the cached counts and ids. Empty
// `tags.open`/`tags.close` leave the state in a disabled mode.
void initializeReasoningState(
    ::llama_context* lctx, ReasoningState& state, ReasoningTags tags);

// Append `tokenStr` to the rolling buffer and flip
// `state.inside_reasoning` when the buffer first contains the
// configured open / close markers. No-op when tags are unset.
void updateReasoningBuffer(const std::string& tokenStr, ReasoningState& state);

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

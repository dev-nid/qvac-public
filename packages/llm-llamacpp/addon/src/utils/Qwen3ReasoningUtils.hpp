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

/// Open / close marker pair used to identify a model's reasoning channel
/// inside the streamed output.
///
/// Different model families use different markers:
///   - Qwen3 family (incl. variants):  `<think>` / `</think>`
///   - Gemma 4 channel format:         `<|channel>thought` / `<channel|>`
///
/// The struct stores raw substrings; per-model token-count metadata that
/// depends on the active tokenizer (multi-token markers, single-token
/// shortcut for the EOS replacement path) is captured separately in
/// `ReasoningState` during `initializeReasoningState`.
struct ReasoningTags {
  std::string_view open;
  std::string_view close;
};

/// Per-generation reasoning detection state.
///
/// One instance lives on the owning context and is reused across the
/// generation loop. `tags` is configured once during context
/// initialisation (via `initializeReasoningState`) and is then read by
/// every `updateReasoningBuffer` call to detect channel transitions.
///
/// When no reasoning support is configured for the active model the
/// `tags` fields are left empty and the detection helpers become
/// no-ops.
struct ReasoningState {
  ReasoningTags tags;

  /// Number of tokens the open marker expands to under the active
  /// tokenizer (cached at init). Required for start-position arithmetic
  /// when the marker tokenises to more than one piece (e.g. Gemma 4's
  /// `<|channel>thought` is multiple BPE tokens). Set to 0 when no
  /// reasoning tags are configured.
  int openTokenCount = 0;

  /// Number of tokens the template-forced reasoning prefix expands to
  /// (i.e. `tags.open` followed by a newline). Used by callers that
  /// need to map a `thinking_forced_open` template suffix back to its
  /// starting KV position. Set to 0 for models whose template does not
  /// force-open the reasoning channel (e.g. Gemma 4 emits the open
  /// marker itself instead).
  int forcedOpenTokenCount = 0;

  /// Single-token id for the close marker when it tokenises to exactly
  /// one token (Qwen3's `</think>`). Used by the EOS-inside-reasoning
  /// replacement path. Remains `LLAMA_TOKEN_NULL` for multi-token close
  /// markers; the replacement path is skipped in that case and the
  /// model closes its own channel.
  llama_token cached_close_tag_token = LLAMA_TOKEN_NULL;
  llama_token cached_newline_token = LLAMA_TOKEN_NULL;

  bool inside_reasoning = false;
  std::string recent_output_buffer;

  static constexpr size_t BUFFER_SIZE = 50;
};

/// Backwards-compatible alias retained for callers that still spell the
/// type `Qwen3ReasoningState`. The generic name `ReasoningState`
/// reflects that the same buffer is now used for Qwen3, Gemma 4 and
/// future families.
using Qwen3ReasoningState = ReasoningState;

/// Initialise `state` for the given tag pair under the active context's
/// tokenizer:
///   * Stores `tags` on the state.
///   * Tokenises `tags.open` and records its token count so capture
///     arithmetic can map a detected open marker back to its starting
///     KV position.
///   * Tokenises `tags.close`. When it expands to a single token, that
///     id is cached for the EOS-inside-reasoning replacement path.
///   * Tokenises `"\n"` and caches its id (used to inject the post-
///     close newlines).
///
/// Empty `tags.open`/`tags.close` leave the state in a disabled mode
/// (no detection / no replacement).
void initializeReasoningState(
    ::llama_context* lctx, ReasoningState& state, ReasoningTags tags);

/// Legacy entry point that configures `state` with Qwen3-family tags
/// (`<think>` / `</think>`). Prefer `initializeReasoningState` paired
/// with `selectReasoningTagsForModel`; this wrapper exists so existing
/// call-sites can compile unchanged during the tag-abstraction
/// migration.
void initializeQwen3ReasoningState(
    ::llama_context* lctx, ReasoningState& state);

/// Append `tokenStr` to `state.recent_output_buffer`, trim to
/// `BUFFER_SIZE`, and flip `state.inside_reasoning` based on whether
/// the configured `state.tags.open` / `state.tags.close` substrings
/// appear in the buffer.
///
/// No-op when `state.tags.open` or `state.tags.close` is empty (i.e.
/// reasoning detection is disabled for the active model). Empty
/// `tokenStr` is ignored.
void updateReasoningBuffer(
    const std::string& tokenStr, ReasoningState& state);

/// Legacy entry point identical to `updateReasoningBuffer`. Kept so
/// existing call-sites compile unchanged.
void updateQwen3ReasoningBuffer(
    const std::string& tokenStr, ReasoningState& state);

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

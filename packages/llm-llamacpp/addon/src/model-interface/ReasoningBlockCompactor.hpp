#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include <llama.h>

#include "../utils/ReasoningRollbackState.hpp"
#include "ToolsCompactController.hpp"

namespace qvac_lib_inference_addon_llama {

// Per-inference reasoning-block compaction lifecycle, shared between
// `TextLlmContext` and `MtmdLlmContext`. Owns:
//
//   * the open/close span (`<think>...</think>`) tracking,
//   * the end-of-prefill snapshot capture (delegated to
//     `ReasoningRollbackState` after a feature-gate check),
//   * the pure-attention `seq_rm + seq_add` compaction path and the
//     recurrent / hybrid full-state restore + replay path,
//   * the `thinkingBlockDiscards` and `thinkingCompactionFailed`
//     runtime stats counters.
//
// State is per-inference. Call `reset()` at the start of each
// `evalMessageWithTools`. Feature flags (`removeThinkingFromContext`,
// `reasoningEnabled`, `needsRecurrentSnapshot`) are set by the owning
// context — they are configured externally because their lifecycles
// (per-request, per-load, per-model) differ and the compactor stays
// agnostic to those.
//
// Position-specific bookkeeping (`nPast_` for text vs `current_.pos /
// .cacheTokens` and `protectedPrefix_` for multimodal) is applied by
// the caller using the returned `Outcome`. The compactor handles only
// the cache-side operations, logging, stats, and tools-compact slide
// notification.
class ReasoningBlockCompactor {
public:
  ReasoningBlockCompactor(
      utils::ReasoningRollbackState& rollback, ToolsCompactController& tools);

  // ---- Feature gates ----
  void setRemoveThinkingFromContext(bool v) noexcept {
    removeThinkingFromContext_ = v;
  }
  [[nodiscard]] bool removeThinkingFromContext() const noexcept {
    return removeThinkingFromContext_;
  }
  void setReasoningEnabled(bool v) noexcept { reasoningEnabled_ = v; }
  void setNeedsRecurrentSnapshot(bool v) noexcept {
    needsRecurrentSnapshot_ = v;
  }
  [[nodiscard]] bool needsRecurrentSnapshot() const noexcept {
    return needsRecurrentSnapshot_;
  }

  // ---- Span tracking ----
  //
  // Single-block policy: only the first `<think>...</think>` of an
  // inference is tracked. Later open markers (no model currently emits
  // them) are ignored.
  void setOpenSpan(llama_pos start);
  [[nodiscard]] bool hasOpenSpan() const noexcept {
    return thinkSpan_.has_value();
  }
  void clearSpan() noexcept {
    thinkSpan_.reset();
    pendingThinkCloseCapture_ = false;
  }

  // ---- Close-marker capture lifecycle ----
  //
  // `requestCloseCapture()` is called when the reasoning detector
  // observes the close marker but the marker token has not yet been
  // committed to the cache. `onCloseCommitted(pos)` is called once the
  // marker has been committed (so `pos` is the cache position after
  // commit); it finalises `thinkSpan_->second` and, on recurrent /
  // hybrid memory, starts post-reasoning token capture for replay.
  void requestCloseCapture() noexcept { pendingThinkCloseCapture_ = true; }
  [[nodiscard]] bool hasPendingCloseCapture() const noexcept {
    return pendingThinkCloseCapture_;
  }
  void onCloseCommitted(llama_pos pos);

  // ---- Post-reasoning token capture (delegates to rollback state) ----
  //
  // No-op when capture is inactive or the token id is null.
  void recordPostReasoningToken(llama_token id) {
    rollback_.recordPostReasoningToken(id);
  }

  // ---- End-of-prefill snapshot ----
  //
  // Captures the full sequence state at `pos` when the feature gates
  // pass (recurrent memory + remove-thinking on + reasoning channel
  // recognised). Logs and bumps `thinkingCompactionFailed_` on
  // capture underflow. `labelTag` is "[TextLlm]" / "[MtmdLlm]" for
  // logs.
  void snapshotAtPrefillBoundary(
      ::llama_context* ctx, llama_seq_id seqId, llama_pos pos,
      const char* labelTag);

  // ---- Compaction ----
  //
  // Performs end-of-generation compaction at the current cache cursor
  // `pos`. The returned `Outcome` carries the new position, the
  // dropped-token count, and the kept-prefix end (used by callers to
  // adjust `firstMsgTokens_` / `protectedPrefix_`). The compactor
  // itself does not write to the caller's position fields.
  //
  // RAII cleanup: per-inference state (`thinkSpan_`, reasoning boundary
  // snapshot, post-reasoning buffer, capture flag) is cleared on every
  // exit so a no-op or failure can't leave stale state behind.
  struct Outcome {
    enum class Kind {
      // Feature off, no span captured, degenerate or overshooting span.
      NoOp,
      CompactedAttention,
      CompactedRecurrent,
      FailedAttention,
      FailedRecurrentRestore,
      FailedRecurrentReplay,
    };
    Kind kind = Kind::NoOp;
    // New cache position the caller should adopt. Unset for `NoOp`.
    llama_pos newPos = 0;
    // Tokens dropped from the cache. `pos - newPos` for the attention
    // path; `pos - newPos` minus the residue for the recurrent path
    // (caller doesn't need to compute this).
    llama_pos discarded = 0;
    // Original span boundaries (for logging or caller-side guards).
    llama_pos spanStart = 0;
    llama_pos spanEnd = 0;
    // First cache position the caller should treat as the new
    // protected-prefix end. Equals `spanStart` for the attention path
    // (the slice is removed and the tail shifts left); equals the
    // restored boundary `nPast` for the recurrent path (the prefix
    // before the boundary is kept verbatim).
    llama_pos keptPrefixEnd = 0;
    // Post-reasoning tokens replayed (recurrent path only).
    size_t replayedTokens = 0;
  };
  [[nodiscard]] Outcome compact(
      ::llama_context* ctx, llama_seq_id seqId, llama_pos pos,
      const char* labelTag);

  // Access to the underlying tools-compact controller. Exposed so
  // `ContextShifter` can route slide notifications to the same
  // controller without holding an independent reference.
  [[nodiscard]] ToolsCompactController& toolsController() noexcept {
    return tools_;
  }

  // ---- Stats ----
  void incrementCompactionFailed() noexcept { ++thinkingCompactionFailed_; }
  [[nodiscard]] int32_t blockDiscards() const noexcept {
    return thinkingBlockDiscards_;
  }
  void resetBlockDiscards() noexcept { thinkingBlockDiscards_ = 0; }
  [[nodiscard]] int32_t compactionFailed() const noexcept {
    return thinkingCompactionFailed_;
  }
  void resetCompactionFailed() noexcept { thinkingCompactionFailed_ = 0; }

  // Per-inference reset of span + close-capture state. Stats and
  // feature flags are NOT reset (stats are managed via dedicated
  // reset methods at the `LlamaModel` level, feature flags are
  // configured externally per request).
  void reset() noexcept {
    thinkSpan_.reset();
    pendingThinkCloseCapture_ = false;
  }

private:
  utils::ReasoningRollbackState& rollback_;
  ToolsCompactController& tools_;

  std::optional<std::pair<llama_pos, llama_pos>> thinkSpan_;
  bool pendingThinkCloseCapture_ = false;

  bool removeThinkingFromContext_ = false;
  bool reasoningEnabled_ = false;
  bool needsRecurrentSnapshot_ = false;

  int32_t thinkingBlockDiscards_ = 0;
  int32_t thinkingCompactionFailed_ = 0;
};

} // namespace qvac_lib_inference_addon_llama

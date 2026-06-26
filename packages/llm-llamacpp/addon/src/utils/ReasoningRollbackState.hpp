#pragma once

#include <cstddef>
#include <vector>

#include <llama.h>

#include "RecurrentStateSnapshot.hpp"

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Shared per-inference rollback state for recurrent / hybrid SSM models.
// Owns the duplicated snapshot lifecycle that previously lived on both
// `TextLlmContext` and `MtmdLlmContext`:
//
//   * a prefill-entry full-state snapshot, restored on cancellation
//     that fires before prefill finishes;
//   * an end-of-prefill full-state snapshot, restored both by
//     thinking-block compaction and by cancellation during generation;
//   * the post-reasoning token capture buffer used to replay the
//     visible answer after restoring the end-of-prefill snapshot.
//
// Failure handling stays in the caller: `capture*` and `restore*`
// return false when the underlying llama.cpp call short-reads, and the
// caller decides how to surface that (typically `++thinkingCompactionFailed_`
// plus a warning log).
//
// Lifetime: per-inference. `reset()` MUST be called at the start of
// each `evalMessageWithTools` so leftover state from a cancelled prior
// turn cannot block a fresh snapshot.
class ReasoningRollbackState {
public:
  // ---- Prefill-entry snapshot (cancel during prefill) ----
  //
  // Captures the full sequence state at `nPast` so a mid-prefill cancel
  // can restore the pre-prefill cursor in one call. Caller should
  // gate on `needsRecurrentSnapshot` first.
  bool capturePrefillEntry(
      ::llama_context* ctx, llama_seq_id seqId, llama_pos nPast);
  // No-op when no snapshot is held. Returns false only when a held
  // snapshot fails to restore.
  bool restorePrefillEntry(::llama_context* ctx, llama_seq_id seqId);
  [[nodiscard]] bool hasPrefillEntry() const noexcept {
    return !prefillEntry_.empty();
  }
  [[nodiscard]] llama_pos prefillEntryNPast() const noexcept {
    return prefillEntry_.nPast;
  }
  void clearPrefillEntry() noexcept { prefillEntry_.clear(); }

  // ---- End-of-prefill snapshot (compaction + cancel during generation) ----
  //
  // No-op if a snapshot already exists for this inference, so the
  // caller doesn't have to re-check before invoking. Returns false on
  // capture failure (the snapshot is cleared in that case).
  bool captureReasoningBoundary(
      ::llama_context* ctx, llama_seq_id seqId, llama_pos nPast);
  // No-op when no snapshot is held. Returns false only when a held
  // snapshot fails to restore.
  bool restoreReasoningBoundary(::llama_context* ctx, llama_seq_id seqId);
  [[nodiscard]] bool hasReasoningBoundary() const noexcept {
    return !reasoningBoundary_.empty();
  }
  [[nodiscard]] llama_pos reasoningBoundaryNPast() const noexcept {
    return reasoningBoundary_.nPast;
  }
  void clearReasoningBoundary() noexcept { reasoningBoundary_.clear(); }

  // ---- Post-reasoning capture (replay buffer) ----
  //
  // Capture is started by the caller once the close marker has been
  // committed AND a reasoning-boundary snapshot exists. Tokens are
  // appended only while capture is active; `recordPostReasoningToken`
  // is a no-op for inactive capture or null token ids.
  void startPostReasoningCapture(bool enable) noexcept {
    capturingPostReasoning_ = enable;
  }
  [[nodiscard]] bool isCapturingPostReasoning() const noexcept {
    return capturingPostReasoning_;
  }
  void recordPostReasoningToken(llama_token id);
  [[nodiscard]] const std::vector<llama_token>&
  postReasoningTokens() const noexcept {
    return postReasoningTokens_;
  }
  [[nodiscard]] size_t postReasoningTokenCount() const noexcept {
    return postReasoningTokens_.size();
  }
  // Truncate the replay buffer to at most `maxSize` tokens. Used when
  // the tools-compact tail trim shrinks the live tail between the
  // close-marker capture and replay.
  void clipPostReasoningTokens(size_t maxSize);
  void clearPostReasoning() noexcept;

  // Replays captured tokens through the decoder, attaching them at
  // positions starting at `reasoningBoundaryNPast()`. Caller should
  // ensure the boundary snapshot was already restored. Returns false
  // if any sub-batch decode call reports a non-zero error.
  bool replayPostReasoning(::llama_context* ctx, llama_seq_id seqId);

  // Clears all per-inference state. Safe to call regardless of which
  // (if any) snapshots are currently held.
  void reset() noexcept;

private:
  RecurrentStateSnapshot prefillEntry_;
  RecurrentStateSnapshot reasoningBoundary_;
  std::vector<llama_token> postReasoningTokens_;
  bool capturingPostReasoning_ = false;
};

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

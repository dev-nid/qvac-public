#include "ReasoningRollbackState.hpp"

#include <cstddef>

#include "RecurrentStateSnapshot.hpp"

namespace qvac_lib_inference_addon_llama {
namespace utils {

bool ReasoningRollbackState::capturePrefillEntry(
    ::llama_context* ctx, llama_seq_id seqId, llama_pos nPast) {
  // Drop any leftover prefill-entry snapshot from a previous request
  // so a failed capture below cannot leave stale bytes accessible to
  // `restorePrefillEntry`.
  prefillEntry_.clear();
  return snapshotRecurrentState(ctx, seqId, nPast, prefillEntry_);
}

bool ReasoningRollbackState::restorePrefillEntry(
    ::llama_context* ctx, llama_seq_id seqId) {
  if (prefillEntry_.empty()) {
    return false;
  }
  return restoreRecurrentState(ctx, seqId, prefillEntry_);
}

bool ReasoningRollbackState::captureReasoningBoundary(
    ::llama_context* ctx, llama_seq_id seqId, llama_pos nPast) {
  if (!reasoningBoundary_.empty()) {
    // Already snapshotted this inference; subsequent calls are no-ops
    // so callers don't have to gate before invoking.
    return true;
  }
  if (!snapshotRecurrentState(ctx, seqId, nPast, reasoningBoundary_)) {
    // Defensive: the primitive clears on short-read, but make sure
    // `hasReasoningBoundary()` cannot accidentally report true after a
    // failed capture.
    reasoningBoundary_.clear();
    return false;
  }
  return true;
}

bool ReasoningRollbackState::restoreReasoningBoundary(
    ::llama_context* ctx, llama_seq_id seqId) {
  if (reasoningBoundary_.empty()) {
    return false;
  }
  return restoreRecurrentState(ctx, seqId, reasoningBoundary_);
}

void ReasoningRollbackState::recordPostReasoningToken(llama_token id) {
  if (!capturingPostReasoning_ || id == LLAMA_TOKEN_NULL) {
    return;
  }
  postReasoningTokens_.push_back(id);
}

void ReasoningRollbackState::appendPostReasoningToken(llama_token id) {
  if (id == LLAMA_TOKEN_NULL) {
    return;
  }
  postReasoningTokens_.push_back(id);
}

void ReasoningRollbackState::clipPostReasoningTokens(size_t maxSize) {
  if (postReasoningTokens_.size() > maxSize) {
    postReasoningTokens_.resize(maxSize);
  }
}

void ReasoningRollbackState::clearPostReasoning() noexcept {
  postReasoningTokens_.clear();
  capturingPostReasoning_ = false;
}

bool ReasoningRollbackState::replayPostReasoning(
    ::llama_context* ctx, llama_seq_id seqId) {
  if (postReasoningTokens_.empty()) {
    return true;
  }
  return replayTokensThroughDecoder(
      ctx, seqId, postReasoningTokens_, reasoningBoundary_.nPast);
}

void ReasoningRollbackState::reset() noexcept {
  prefillEntry_.clear();
  reasoningBoundary_.clear();
  postReasoningTokens_.clear();
  capturingPostReasoning_ = false;
}

void ReasoningRollbackState::seedReasoningBoundaryForTesting(
    std::vector<uint8_t> data, llama_pos nPast) noexcept {
  reasoningBoundary_.data = std::move(data);
  reasoningBoundary_.nPast = nPast;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

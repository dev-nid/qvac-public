#pragma once

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Recurrent / hybrid compaction restores an end-of-prefill snapshot and
// replays the post-reasoning tail. This strategy is only valid when:
//   * `thinkingForcedOpen` — the chat template already force-opened the
//     reasoning block during prefill, so the restored prefix contains the
//     matching opener.
//   * `closeMarkerSingleToken` — the reasoning close tag tokenises to a
//     single token. The replay path seeds `postReasoningTokens_` with the
//     one sampled token that triggers the close-detection flip in
//     `updateReasoningBuffer`; a multi-piece close would leave the SSM with
//     an unbalanced `<think>` opener followed by only the tail piece.
//
// When `remove_thinking_from_context` is enabled for recurrent / hybrid
// memory and reasoning is active, unsupported templates must hard-fail
// instead of silently preserving reasoning in cache. `Disabled` means the
// policy is irrelevant for this request (pure attention, feature off, or no
// active reasoning channel); the two `Unsupported*` states mean callers
// should surface a StatusError after any required rollback.
enum class RecurrentReasoningBoundaryDecision {
  Disabled,
  Capture,
  UnsupportedGeneratedOpener,
  UnsupportedMultiTokenClose,
};

[[nodiscard]] inline RecurrentReasoningBoundaryDecision
recurrentReasoningBoundaryDecision(
    bool needsRecurrentSnapshot, bool removeThinkingFromContext,
    bool reasoningEnabled, bool thinkingForcedOpen,
    bool closeMarkerSingleToken) noexcept {
  if (!needsRecurrentSnapshot || !removeThinkingFromContext ||
      !reasoningEnabled) {
    return RecurrentReasoningBoundaryDecision::Disabled;
  }
  if (!thinkingForcedOpen) {
    return RecurrentReasoningBoundaryDecision::UnsupportedGeneratedOpener;
  }
  if (!closeMarkerSingleToken) {
    return RecurrentReasoningBoundaryDecision::UnsupportedMultiTokenClose;
  }
  return RecurrentReasoningBoundaryDecision::Capture;
}

[[nodiscard]] inline const char* recurrentReasoningBoundaryFailureReason(
    RecurrentReasoningBoundaryDecision decision) noexcept {
  switch (decision) {
  case RecurrentReasoningBoundaryDecision::UnsupportedGeneratedOpener:
    return "the chat template did not force-open reasoning during prefill";
  case RecurrentReasoningBoundaryDecision::UnsupportedMultiTokenClose:
    return "the reasoning close marker is not a single token";
  case RecurrentReasoningBoundaryDecision::Disabled:
  case RecurrentReasoningBoundaryDecision::Capture:
    return "";
  }
  return "";
}

[[nodiscard]] inline bool shouldCaptureRecurrentReasoningBoundary(
    bool needsRecurrentSnapshot, bool removeThinkingFromContext,
    bool reasoningEnabled, bool thinkingForcedOpen,
    bool closeMarkerSingleToken) noexcept {
  return recurrentReasoningBoundaryDecision(
             needsRecurrentSnapshot,
             removeThinkingFromContext,
             reasoningEnabled,
             thinkingForcedOpen,
             closeMarkerSingleToken) ==
         RecurrentReasoningBoundaryDecision::Capture;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

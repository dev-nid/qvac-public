#pragma once

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Recurrent / hybrid compaction restores an end-of-prefill snapshot and
// replays the post-reasoning tail. Preconditions:
//   * `thinkingForcedOpen` — the chat template already force-opened the
//     reasoning block during prefill, so the restored prefix contains the
//     matching opener.
//   * `closeMarkerSingleToken` — the reasoning close tag tokenises to a
//     single token. The replay path seeds `postReasoningTokens_` with the
//     one sampled token that triggers the close-detection flip in
//     `updateReasoningBuffer`; a multi-piece close would leave the SSM with
//     an unbalanced `<think>` opener followed by only the tail piece.
[[nodiscard]] inline bool shouldCaptureRecurrentReasoningBoundary(
    bool needsRecurrentSnapshot, bool removeThinkingFromContext,
    bool reasoningEnabled, bool thinkingForcedOpen,
    bool closeMarkerSingleToken) noexcept {
  return needsRecurrentSnapshot && removeThinkingFromContext &&
         reasoningEnabled && thinkingForcedOpen && closeMarkerSingleToken;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

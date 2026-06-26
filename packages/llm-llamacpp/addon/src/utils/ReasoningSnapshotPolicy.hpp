#pragma once

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Recurrent / hybrid compaction restores an end-of-prefill snapshot and
// replays the post-reasoning tail. That snapshot is only structurally
// valid when the chat template already force-opened the reasoning block
// during prefill, so the restored prefix contains the matching opener.
[[nodiscard]] inline bool shouldCaptureRecurrentReasoningBoundary(
    bool needsRecurrentSnapshot, bool removeThinkingFromContext,
    bool reasoningEnabled, bool thinkingForcedOpen) noexcept {
  return needsRecurrentSnapshot && removeThinkingFromContext &&
      reasoningEnabled && thinkingForcedOpen;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

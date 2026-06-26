#include <optional>

#include <gtest/gtest.h>
#include <llama.h>

#include "model-interface/ReasoningBlockCompactor.hpp"
#include "model-interface/ToolsCompactController.hpp"
#include "utils/ReasoningRollbackState.hpp"

using qvac_lib_inference_addon_llama::ReasoningBlockCompactor;
using qvac_lib_inference_addon_llama::utils::ReasoningRollbackState;

// Unit coverage for the hybrid / recurrent close-marker replay seam.
//
// End-of-prefill snapshots keep the forced `<think>\n` opener in the
// restored SSM state, so the post-reasoning replay buffer must
// explicitly carry the matching close marker to leave a balanced
// `<think>...</think>` shape after compaction. These tests pin:
//   1. the unconditional append primitive (`appendPostReasoningToken`),
//   2. the compactor wrapper (`recordCloseMarkerForReplay`) feature gates,
//   3. the success path against a seeded boundary snapshot.

TEST(ReasoningRollbackStateAppend, AppendsRegardlessOfCaptureFlag) {
  ReasoningRollbackState rollback;
  EXPECT_FALSE(rollback.isCapturingPostReasoning());
  rollback.appendPostReasoningToken(42);
  rollback.appendPostReasoningToken(7);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 2u);
  EXPECT_EQ(rollback.seededPostReasoningCount(), 2u);
  EXPECT_EQ(rollback.postReasoningTokens()[0], 42);
  EXPECT_EQ(rollback.postReasoningTokens()[1], 7);
}

TEST(ReasoningRollbackStateAppend, SkipsNullToken) {
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(LLAMA_TOKEN_NULL);
  EXPECT_EQ(rollback.postReasoningTokenCount(), 0u);
  EXPECT_EQ(rollback.seededPostReasoningCount(), 0u);
}

TEST(ReasoningRollbackStateAppend, PreservesOrderWithCapturedTokens) {
  // The close marker is seeded via `appendPostReasoningToken` BEFORE
  // capture flips on; everything sampled after lands via
  // `recordPostReasoningToken`. The replay must concatenate them in
  // [close-marker, ...post-close] order so the SSM advance is balanced.
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(/*closeMarker=*/100);
  rollback.startPostReasoningCapture(true);
  rollback.recordPostReasoningToken(/*newline=*/198);
  rollback.recordPostReasoningToken(/*answer=*/2500);

  ASSERT_EQ(rollback.postReasoningTokenCount(), 3u);
  EXPECT_EQ(rollback.postReasoningTokens()[0], 100);
  EXPECT_EQ(rollback.postReasoningTokens()[1], 198);
  EXPECT_EQ(rollback.postReasoningTokens()[2], 2500);
}

TEST(ReasoningRollbackStateAppend, ResetClearsBuffer) {
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(1);
  rollback.appendPostReasoningToken(2);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 2u);
  rollback.reset();
  EXPECT_EQ(rollback.postReasoningTokenCount(), 0u);
  EXPECT_EQ(rollback.seededPostReasoningCount(), 0u);
}

TEST(
    ReasoningRollbackStateClip,
    PreservesSeededCloseMarkerWithEmptyCapturedTail) {
  // `compact()` passes `pos - end` as the captured-tail cap. When no
  // post-close tokens were sampled (e.g. EOS hit immediately after
  // `</think>`), that cap is zero. The seeded close marker MUST
  // survive — dropping it would replay an unbalanced
  // `<think>\n` + answer-tail recurrent state on the next turn.
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(/*closeMarker=*/100);
  ASSERT_EQ(rollback.seededPostReasoningCount(), 1u);

  rollback.clipPostReasoningTokens(/*maxCapturedTail=*/0);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 1u);
  EXPECT_EQ(rollback.postReasoningTokens()[0], 100);
}

TEST(ReasoningRollbackStateClip, PreservesMultipleSeededStructuralTokens) {
  // Future callers may seed more than one structural token before
  // capture flips on. Clipping with an empty live tail must preserve
  // the entire seeded prefix, not just the first token.
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(/*closeMarker=*/100);
  rollback.appendPostReasoningToken(/*structuralNewline=*/198);
  rollback.startPostReasoningCapture(true);
  rollback.recordPostReasoningToken(/*capturedTail=*/2500);
  ASSERT_EQ(rollback.seededPostReasoningCount(), 2u);

  rollback.clipPostReasoningTokens(/*maxCapturedTail=*/0);

  ASSERT_EQ(rollback.postReasoningTokenCount(), 2u);
  EXPECT_EQ(rollback.postReasoningTokens()[0], 100);
  EXPECT_EQ(rollback.postReasoningTokens()[1], 198);
}

TEST(ReasoningRollbackStateClip, KeepsSeededPrefixAndCapsCapturedTail) {
  // Replay buffer is [close_marker, t0, t1, t2]. Live cache only has
  // two post-close tokens left (tools-compact trimmed one). Clip cap
  // is the captured-tail length (2), not the total. The close marker
  // stays; only the last captured token is dropped.
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(/*closeMarker=*/100);
  rollback.startPostReasoningCapture(true);
  rollback.recordPostReasoningToken(/*t0=*/198);
  rollback.recordPostReasoningToken(/*t1=*/2500);
  rollback.recordPostReasoningToken(/*t2=*/9999);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 4u);

  rollback.clipPostReasoningTokens(/*maxCapturedTail=*/2);

  ASSERT_EQ(rollback.postReasoningTokenCount(), 3u);
  EXPECT_EQ(rollback.postReasoningTokens()[0], 100);
  EXPECT_EQ(rollback.postReasoningTokens()[1], 198);
  EXPECT_EQ(rollback.postReasoningTokens()[2], 2500);
}

TEST(ReasoningRollbackStateClip, ClipsAllCapturedTokensWhenNoSeededPrefix) {
  // Baseline for the old shape: if the buffer has only captured tail
  // tokens, cap 0 still means drop everything.
  ReasoningRollbackState rollback;
  rollback.startPostReasoningCapture(true);
  rollback.recordPostReasoningToken(1);
  rollback.recordPostReasoningToken(2);
  ASSERT_EQ(rollback.seededPostReasoningCount(), 0u);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 2u);

  rollback.clipPostReasoningTokens(/*maxCapturedTail=*/0);

  EXPECT_EQ(rollback.postReasoningTokenCount(), 0u);
}

TEST(ReasoningRollbackStateClip, ClearPostReasoningResetsSeededCount) {
  // Seeded count must follow the buffer lifecycle: a fresh inference
  // (post-clear) must not see a stale count that would let the next
  // clip preserve nonexistent tokens.
  ReasoningRollbackState rollback;
  rollback.appendPostReasoningToken(100);
  ASSERT_EQ(rollback.seededPostReasoningCount(), 1u);
  rollback.clearPostReasoning();
  EXPECT_EQ(rollback.seededPostReasoningCount(), 0u);

  rollback.startPostReasoningCapture(true);
  rollback.recordPostReasoningToken(1);
  rollback.recordPostReasoningToken(2);
  ASSERT_EQ(rollback.postReasoningTokenCount(), 2u);
  rollback.clipPostReasoningTokens(/*maxCapturedTail=*/0);
  EXPECT_EQ(rollback.postReasoningTokenCount(), 0u);
}

namespace {

// Helper that wires up the compactor with the gates exposed by its
// public API. The boundary snapshot is left empty by default; callers
// that need `hasReasoningBoundary()` to be true seed it directly via
// `RecurrentStateSnapshot.data`.
struct CompactorFixture {
  ReasoningRollbackState rollback;
  // Tools-compact controller is unused by `recordCloseMarkerForReplay`
  // — its slide notifier only fires from `compact()`. Constructed with
  // an empty profile so it stays in its disabled state for the lifetime
  // of the fixture.
  ToolsCompactController tools{std::nullopt};
  ReasoningBlockCompactor compactor{rollback, tools};
};

} // namespace

TEST(ReasoningBlockCompactorReplaySeed, NoOpWhenRemoveThinkingOff) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(false);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  fx.compactor.recordCloseMarkerForReplay(/*closeMarker=*/42);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
}

TEST(ReasoningBlockCompactorReplaySeed, NoOpWhenReasoningDisabled) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(false);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  fx.compactor.recordCloseMarkerForReplay(42);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
}

TEST(ReasoningBlockCompactorReplaySeed, NoOpForPureAttentionModels) {
  // The replay buffer is consumed only on the recurrent / hybrid
  // compact path. Pure-attention models use `seq_rm + seq_add` and
  // never replay tokens, so seeding the buffer would be dead state.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(false);
  fx.compactor.recordCloseMarkerForReplay(42);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
}

TEST(ReasoningBlockCompactorReplaySeed, NoOpWhenBoundaryNotCaptured) {
  // All feature gates open but no end-of-prefill snapshot exists
  // (e.g. capture underflowed). Recording the close marker would be
  // unsafe — there is nothing to restore to, so compaction will be
  // skipped anyway. The seed must not silently accumulate in that case.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  ASSERT_FALSE(fx.rollback.hasReasoningBoundary());
  fx.compactor.recordCloseMarkerForReplay(42);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
}

TEST(ReasoningBlockCompactorReplaySeed, AppendsWhenAllGatesAndBoundaryPresent) {
  // Simulate a successful end-of-prefill capture by seeding a non-empty
  // snapshot payload directly. The compactor only needs
  // `hasReasoningBoundary()` to be true to know the recurrent restore
  // path is viable — the snapshot's exact bytes are irrelevant for
  // seeding the replay buffer.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  fx.rollback.seedReasoningBoundaryForTesting({0xAB, 0xCD}, /*nPast=*/10);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  fx.compactor.recordCloseMarkerForReplay(/*closeMarker=*/123);
  ASSERT_EQ(fx.rollback.postReasoningTokenCount(), 1u);
  EXPECT_EQ(fx.rollback.seededPostReasoningCount(), 1u);
  EXPECT_EQ(fx.rollback.postReasoningTokens()[0], 123);
}

TEST(ReasoningBlockCompactorReplaySeed, SkipsNullCloseMarker) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  fx.rollback.seedReasoningBoundaryForTesting({0xAB}, /*nPast=*/5);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  fx.compactor.recordCloseMarkerForReplay(LLAMA_TOKEN_NULL);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
  EXPECT_EQ(fx.rollback.seededPostReasoningCount(), 0u);
}

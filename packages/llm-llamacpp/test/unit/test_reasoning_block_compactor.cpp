#include <optional>

#include <gtest/gtest.h>
#include <inference-addon-cpp/Errors.hpp>
#include <llama.h>

#include "model-interface/ReasoningBlockCompactor.hpp"
#include "model-interface/ToolsCompactController.hpp"
#include "utils/ReasoningRollbackState.hpp"
#include "utils/ReasoningSnapshotPolicy.hpp"

using qvac_lib_inference_addon_llama::ReasoningBlockCompactor;
using qvac_lib_inference_addon_llama::utils::ReasoningRollbackState;
using qvac_lib_inference_addon_llama::utils::
    shouldCaptureRecurrentReasoningBoundary;

// Unit coverage for the hybrid / recurrent close-marker replay seam.
//
// End-of-prefill snapshots keep the forced `<think>\n` opener in the
// restored SSM state, so the post-reasoning replay buffer must
// explicitly carry the matching close marker to leave a balanced
// `<think>...</think>` shape after compaction. These tests pin:
//   1. the unconditional append primitive (`appendPostReasoningToken`),
//   2. the compactor wrapper (`recordCloseMarkerForReplay`) feature gates,
//   3. the success path against a seeded boundary snapshot.

TEST(ReasoningSnapshotPolicy, CapturesOnlyForForcedOpenRecurrentReasoning) {
  EXPECT_TRUE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true));
}

TEST(ReasoningSnapshotPolicy, SkipsGeneratedOpenRecurrentReasoning) {
  // Generated-opener recurrent turns cannot use an end-of-prefill
  // snapshot: the restored prefix would not contain `<think>`, so
  // replaying `</think>` would poison the next recurrent state.
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/false));
}

TEST(ReasoningSnapshotPolicy, SkipsWhenFeatureOrReasoningGateIsClosed) {
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/false,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true));
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/false,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true));
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/false,
      /*thinkingForcedOpen=*/true));
}

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
// that need `hasReasoningBoundary()` to be true seed a sentinel
// file-backed snapshot through the rollback test seam.
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
  fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/10);
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
  fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/5);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  fx.compactor.recordCloseMarkerForReplay(LLAMA_TOKEN_NULL);
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
  EXPECT_EQ(fx.rollback.seededPostReasoningCount(), 0u);
}

// Pin the close-capture handshake contract used by every close-marker
// site (normal buffer-transition path AND EOS-substitution path):
// `onCloseCommitted` only records the span end after a prior
// `requestCloseCapture()`. The EOS-substitution path in
// `TextLlmContext::handleReasoningEOS` previously called
// `onCloseCommitted` directly without flipping the flag, which silently
// dropped the close position and left `compactThinkSpan` to bail at
// `end < 0`. These tests document the contract so any future caller
// regression surfaces here rather than as a "multi-turn compaction
// quietly stops working" integration failure.
TEST(ReasoningBlockCompactorCloseCommit, IsNoOpWithoutPriorRequest) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setOpenSpan(/*start=*/10);
  ASSERT_TRUE(fx.compactor.hasOpenSpan());
  ASSERT_FALSE(fx.compactor.hasPendingCloseCapture());

  // No `requestCloseCapture()` ahead of this — the flag never flipped,
  // so the commit is dropped and the span end stays unset.
  fx.compactor.onCloseCommitted(/*pos=*/42);
  EXPECT_FALSE(fx.compactor.hasCapturedCloseSpanForTesting());
}

TEST(ReasoningBlockCompactorCloseCommit, RecordsSpanEndAfterRequest) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setOpenSpan(/*start=*/10);
  ASSERT_TRUE(fx.compactor.hasOpenSpan());

  fx.compactor.requestCloseCapture();
  ASSERT_TRUE(fx.compactor.hasPendingCloseCapture());
  fx.compactor.onCloseCommitted(/*pos=*/42);
  EXPECT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());
  EXPECT_FALSE(fx.compactor.hasPendingCloseCapture());
}

// ============================================================================
// Failure contract — hybrid restore/replay
// ============================================================================
//
// A `state_seq_load_file` short-read on the boundary restore, or a
// `llama_decode` failure during the post-reasoning replay, leaves live
// KV/SSM memory in a state that no longer matches any cursor the caller
// might synthesise. The compactor no longer papers over this with a
// "best-effort" retained-state Outcome (that path silently corrupted
// the next turn's decode and any concurrent saveCache). Instead:
//
//   1. `thinkingCompactionFailed_` increments by exactly 1 per failure.
//   2. `thinkingBlockDiscards_` does NOT increment (no successful drop).
//   3. The affected sequence is best-effort wiped via
//      `llama_memory_seq_rm` (skipped when `ctx == nullptr` in unit
//      tests; verified end-to-end by the driver-level tests that
//      assert next-request-starts-clean).
//   4. `compact()` throws `qvac_errors::StatusError`, so callers must
//      catch, reset their positional accounting to zero (matching the
//      wiped sequence), and re-throw. No saveCache path can serialize
//      a header whose metadata contradicts memory, because the throw
//      unwinds past every save site.
//
// We force the failure via `ctx == nullptr`: `restoreRecurrentState`
// is the first call inside the recurrent branch and short-returns
// false on null `lctx`. The symmetric replay branch shares the exact
// same throw + wipe shape (see the replay `throw` site in
// `ReasoningBlockCompactor.cpp`) but is unreachable from this fixture
// without either a test seam on `replayPostReasoning` or a real
// `llama_context` — restore trips first. We pin the shape here and
// leave replay as a code-reading equivalence rather than introducing
// a production seam solely for parity coverage.

TEST(
    ReasoningBlockCompactorFailureStats,
    RestoreFailureThrowsIncrementsCompactionFailedAndClearsInternalState) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);

  constexpr llama_pos kSnapshotPos = 10;
  constexpr llama_pos kSpanStart = 15;
  constexpr llama_pos kSpanEnd = 20;
  constexpr llama_pos kLivePos = 25;

  // Seed a captured boundary so the recurrent branch is actually
  // entered (the "no boundary captured" early-return is its own
  // outcome — see `NoOpWhenBoundaryNotCaptured` above).
  fx.rollback.seedReasoningBoundaryForTesting(kSnapshotPos);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  // A fully committed span — open + close + still inside the live cache.
  fx.compactor.setOpenSpan(kSpanStart);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(kSpanEnd);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  ASSERT_EQ(fx.compactor.compactionFailed(), 0);
  ASSERT_EQ(fx.compactor.blockDiscards(), 0);

  // `ctx == nullptr` -> `restoreRecurrentState` returns false ->
  // `restoreReasoningBoundary` returns false -> `compact()` throws.
  EXPECT_THROW(
      {
        (void)fx.compactor.compact(
            /*ctx=*/nullptr, /*seqId=*/0, kLivePos, "[Test]");
      },
      qvac_errors::StatusError);

  // Stats: exactly one failure recorded, no success counted.
  EXPECT_EQ(fx.compactor.compactionFailed(), 1);
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);

  // The `ResetGuard` in `compact()` runs on the exception path too,
  // so per-inference state (span, boundary snapshot, replay buffer)
  // must be fully cleared. Without this the next inference's
  // `snapshotAtPrefillBoundary` no-ops on the stale boundary and the
  // driver would replay stale post-reasoning tokens.
  EXPECT_FALSE(fx.compactor.hasOpenSpan());
  EXPECT_FALSE(fx.rollback.hasReasoningBoundary());
  EXPECT_EQ(fx.rollback.postReasoningTokenCount(), 0u);
}

TEST(
    ReasoningBlockCompactorFailureStats,
    NextCompactAfterRestoreFailureIsCleanNoOp) {
  // The reviewer's "next request starts from a clean/reset state"
  // invariant, exercised at the compactor level: after a throw, a
  // fresh compact() on the same instance MUST not carry over the
  // failed inference's span or boundary. If the ResetGuard ever
  // regressed and left `thinkSpan_` behind, this compact would take
  // the pure-attention path with a stale `spanStart`/`spanEnd` and
  // either bump `thinkingBlockDiscards_` for a phantom drop or
  // re-throw against the same null ctx.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);

  fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/10);
  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/20);
  EXPECT_THROW(
      {
        (void)fx.compactor.compact(
            /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
      },
      qvac_errors::StatusError);
  ASSERT_EQ(fx.compactor.compactionFailed(), 1);

  // Simulating "next turn": no new span, no seeded boundary.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/0, "[Test]");
  EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
  EXPECT_EQ(fx.compactor.compactionFailed(), 1);
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);
}

TEST(
    ReasoningBlockCompactorFailureStats,
    NoOpOutcomesDoNotIncrementCompactionFailed) {
  // Non-failure no-op paths (degenerate span where close never landed,
  // and recurrent runs with no boundary captured) leave the cache
  // untouched and MUST NOT bump the failure counter. Without this
  // guard, any caller skipping `compactThinkSpan` (e.g. a turn where
  // reasoning never closed) would silently inflate
  // `thinkingCompactionFailed` and look like a real failure to
  // dashboards. Distinct from the throw-and-clear contract in the
  // hybrid restore/replay failure tests above.
  //
  // We deliberately do NOT cover the `end <= start` and `end > pos`
  // sub-cases here because the public API does not expose a way to
  // construct those configurations: `setOpenSpan` rejects `start < 0`
  // and `onCloseCommitted` only ever monotonically writes `end`. Those
  // bail-outs are defensive backstops that fire only if the compactor
  // is mis-used by an internal caller.

  // 1. Degenerate span (close never committed -> end < 0).
  {
    CompactorFixture fx;
    fx.compactor.setRemoveThinkingFromContext(true);
    fx.compactor.setReasoningEnabled(true);
    fx.compactor.setNeedsRecurrentSnapshot(true);
    fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/10);
    fx.compactor.setOpenSpan(/*start=*/15);
    // No requestCloseCapture / onCloseCommitted -> end stays -1.
    ASSERT_FALSE(fx.compactor.hasCapturedCloseSpanForTesting());

    const auto outcome = fx.compactor.compact(
        /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
    EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
    EXPECT_EQ(fx.compactor.compactionFailed(), 0);
  }

  // 2. Recurrent path with no boundary captured -> NoOp, not a failure.
  {
    CompactorFixture fx;
    fx.compactor.setRemoveThinkingFromContext(true);
    fx.compactor.setReasoningEnabled(true);
    fx.compactor.setNeedsRecurrentSnapshot(true);
    ASSERT_FALSE(fx.rollback.hasReasoningBoundary());
    fx.compactor.setOpenSpan(/*start=*/15);
    fx.compactor.requestCloseCapture();
    fx.compactor.onCloseCommitted(/*pos=*/20);

    const auto outcome = fx.compactor.compact(
        /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
    EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
    EXPECT_EQ(fx.compactor.compactionFailed(), 0);
  }
}

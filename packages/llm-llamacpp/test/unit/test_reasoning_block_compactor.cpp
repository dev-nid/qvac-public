#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include <inference-addon-cpp/Errors.hpp>
#include <llama.h>

#include "model-interface/ContextSlider.hpp"
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
      /*thinkingForcedOpen=*/true,
      /*closeMarkerSingleToken=*/true));
}

TEST(ReasoningSnapshotPolicy, SkipsGeneratedOpenRecurrentReasoning) {
  // Generated-opener recurrent turns cannot use an end-of-prefill
  // snapshot: the restored prefix would not contain `<think>`, so
  // replaying `</think>` would poison the next recurrent state.
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/false,
      /*closeMarkerSingleToken=*/true));
}

TEST(ReasoningSnapshotPolicy, SkipsWhenFeatureOrReasoningGateIsClosed) {
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/false,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true,
      /*closeMarkerSingleToken=*/true));
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/false,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true,
      /*closeMarkerSingleToken=*/true));
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/false,
      /*thinkingForcedOpen=*/true,
      /*closeMarkerSingleToken=*/true));
}

// Recurrent replay seeds `postReasoningTokens_` with the single sampled
// token that flips `updateReasoningBuffer` out of `inside_reasoning`. If
// the close tag tokenises to more than one piece, that seed captures
// only the tail piece and the restored SSM state ends with an unbalanced
// `<think>` opener. The policy MUST reject the boundary snapshot in that
// case so `remove_thinking_from_context` degrades to leaving reasoning
// tokens in the cache instead of silently corrupting recurrent state.
TEST(ReasoningSnapshotPolicy, SkipsWhenCloseMarkerIsMultiToken) {
  EXPECT_FALSE(shouldCaptureRecurrentReasoningBoundary(
      /*needsRecurrentSnapshot=*/true,
      /*removeThinkingFromContext=*/true,
      /*reasoningEnabled=*/true,
      /*thinkingForcedOpen=*/true,
      /*closeMarkerSingleToken=*/false));
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
// Failure contract — uniform hard-fail
// ============================================================================
//
// Any inability to remove the reasoning span from cache is a hard
// failure under the default-on `remove_thinking_from_context` contract
// (PR #2813). `snapshotAtPrefillBoundary` still throws
// `qvac_errors::StatusError` on boundary-capture failure (recovery
// happens one level up in `snapshotForRecurrentRollback`), but
// `compact()` reports failures via `Outcome::Kind::FailedKvIntact` /
// `Outcome::Kind::FailedKvWiped` so callers can choose the correct
// live-KV recovery (pre-request rollback vs full reset) before
// rethrowing. In every failure path `thinkingBlockDiscards` never
// bumps for the failed drop.
//
// Coverage:
//   * Boundary-capture failure (`snapshotAtPrefillBoundary` on
//     `ctx == nullptr`, which short-reads inside
//     `captureReasoningBoundary`).
//   * Hybrid restore failure (`compact()` on `ctx == nullptr` with a
//     seeded boundary) reports `FailedKvWiped`.
//   * Generated-opener recurrent turn — hybrid model, no boundary
//     snapshot captured — must cleanly no-op (span never opens on
//     recurrent+no-boundary; `compact()` returns `NoOp`).
//   * Non-failure no-op paths do NOT throw and do NOT bump discards.
//
// The symmetric replay throw shape is exercised end-to-end by the
// driver-level integration tests; the compactor unit fixture cannot
// reach it without either a test seam on `replayPostReasoning` or a
// real `llama_context`.

TEST(
    ReasoningBlockCompactorFailureStats,
    BoundaryCaptureFailureThrowsAndLeavesNoStaleState) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);

  // `ctx == nullptr` short-circuits `captureReasoningBoundary` to
  // return false, which now throws under the hard-fail contract.
  ASSERT_FALSE(fx.rollback.hasReasoningBoundary());
  EXPECT_THROW(
      {
        fx.compactor.snapshotAtPrefillBoundary(
            /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/10, "[Test]");
      },
      qvac_errors::StatusError);

  // No spurious boundary or discard bookkeeping on failure.
  EXPECT_FALSE(fx.rollback.hasReasoningBoundary());
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);
}

TEST(
    ReasoningBlockCompactorFailureStats,
    RestoreFailureThrowsAndClearsInternalState) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);

  constexpr llama_pos kSnapshotPos = 10;
  constexpr llama_pos kSpanStart = 15;
  constexpr llama_pos kSpanEnd = 20;
  constexpr llama_pos kLivePos = 25;

  fx.rollback.seedReasoningBoundaryForTesting(kSnapshotPos);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  fx.compactor.setOpenSpan(kSpanStart);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(kSpanEnd);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  ASSERT_EQ(fx.compactor.blockDiscards(), 0);

  // `ctx == nullptr` -> `restoreRecurrentState` returns false ->
  // `restoreReasoningBoundary` returns false -> `compact()` reports
  // `FailedKvWiped` with a populated failureMessage so the caller can
  // rethrow with matching context.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, kLivePos, "[Test]");
  EXPECT_EQ(
      outcome.kind, ReasoningBlockCompactor::Outcome::Kind::FailedKvWiped);
  EXPECT_FALSE(outcome.failureMessage.empty());

  // No successful drop counted.
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);

  // The `ResetGuard` in `compact()` runs on the failure path too,
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
  // invariant, exercised at the compactor level: after a failure
  // outcome, a fresh compact() on the same instance MUST not carry
  // over the failed inference's span or boundary.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);

  fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/10);
  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/20);
  const auto failed = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
  ASSERT_EQ(failed.kind, ReasoningBlockCompactor::Outcome::Kind::FailedKvWiped);

  // Simulating "next turn": no new span, no seeded boundary.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/0, "[Test]");
  EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);
}

// Generated-opener recurrent regression (PR #2813 review): a hybrid
// model with no boundary snapshot (template does not force-open
// `<think>`, so `ReasoningSnapshotPolicy` skipped the snapshot) whose
// model then emits `<think>...</think>` during decode must not open a
// span — otherwise `compact()` hits its defensive no-boundary branch
// and wipes the sequence instead of cleanly no-oping.
TEST(
    ReasoningBlockCompactorFailureStats,
    HybridGeneratedOpenerRecurrentSpanSkipsCompactionAsNoOp) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true);
  ASSERT_FALSE(fx.rollback.hasReasoningBoundary());

  fx.compactor.setOpenSpan(/*start=*/15);
  EXPECT_FALSE(fx.compactor.hasOpenSpan())
      << "recurrent + no boundary must not record a span — otherwise "
         "compact() will hit its defensive branch and wipe the sequence";

  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/20);
  EXPECT_FALSE(fx.compactor.hasCapturedCloseSpanForTesting());

  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
  EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp)
      << "generated-opener recurrent compaction must be a clean no-op, "
         "not FailedKvWiped";
  EXPECT_TRUE(outcome.failureMessage.empty());
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);
}

TEST(ReasoningBlockCompactorFailureStats, NoOpOutcomesDoNotThrow) {
  // Non-failure no-op paths (degenerate span where close never landed)
  // leave the cache untouched and MUST NOT throw. Without this guard,
  // any caller invoking `compact()` on an incomplete span (e.g. a turn
  // where reasoning never closed) would be spuriously failed.
  //
  // We deliberately do NOT cover the `end <= start` and `end > pos`
  // sub-cases here because the public API does not expose a way to
  // construct those configurations: `setOpenSpan` rejects `start < 0`
  // and `onCloseCommitted` only ever monotonically writes `end`. Those
  // bail-outs are defensive backstops that fire only if the compactor
  // is mis-used by an internal caller.

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
  EXPECT_EQ(fx.compactor.blockDiscards(), 0);
}

namespace {

// Minimal `IContextSliderOps` fakes for the compactor tests.
// `compactKvRange` on the pure-attention path is the only production
// call site the compactor routes through the injectable ops. Two
// fakes are provided so tests can drive either half of the primitive
// contract without a real llama context:
//
//   * `AcceptingSliderOps` — `seqRm` returns `true`, so the compactor
//     proceeds to `seqAdd` and reports `CompactedAttention`. Used by
//     the successful-drop tests to observe that `seqAdd` fires and
//     that side-effect notifications (e.g. `tools_.onSlide`) run.
//   * `RejectingSliderOps` — `seqRm` returns `false` to mimic a
//     rejected primitive. The production contract is "all-or-nothing
//     on rejection", so `seqAdd` MUST NOT fire afterwards; otherwise
//     the compactor's `FailedKvIntact` outcome would be misleading
//     (it would imply KV was touched anyway).
class AcceptingSliderOps final : public IContextSliderOps {
public:
  llama_pos nCtx(llama_context*) const override { return 4096; }

  ContextSliderMemoryHandle memory(llama_context*) const override {
    return fakeMemory_;
  }

  bool seqRm(ContextSliderMemoryHandle, llama_seq_id, llama_pos, llama_pos)
      const override {
    ++seqRmCalls_;
    return true;
  }

  void seqAdd(
      ContextSliderMemoryHandle, llama_seq_id, llama_pos, llama_pos,
      llama_pos) const override {
    ++seqAddCalls_;
  }

  int seqRmCalls() const { return seqRmCalls_; }
  int seqAddCalls() const { return seqAddCalls_; }

private:
  ContextSliderMemoryHandle fakeMemory_ =
      reinterpret_cast<ContextSliderMemoryHandle>(static_cast<uintptr_t>(0x1));
  mutable int seqRmCalls_ = 0;
  mutable int seqAddCalls_ = 0;
};

class RejectingSliderOps final : public IContextSliderOps {
public:
  llama_pos nCtx(llama_context*) const override { return 4096; }

  ContextSliderMemoryHandle memory(llama_context*) const override {
    return fakeMemory_;
  }

  bool seqRm(ContextSliderMemoryHandle, llama_seq_id, llama_pos, llama_pos)
      const override {
    ++seqRmCalls_;
    return false;
  }

  void seqAdd(
      ContextSliderMemoryHandle, llama_seq_id, llama_pos, llama_pos,
      llama_pos) const override {
    ++seqAddCalls_;
  }

  int seqRmCalls() const { return seqRmCalls_; }
  int seqAddCalls() const { return seqAddCalls_; }

private:
  ContextSliderMemoryHandle fakeMemory_ =
      reinterpret_cast<ContextSliderMemoryHandle>(static_cast<uintptr_t>(0x1));
  mutable int seqRmCalls_ = 0;
  mutable int seqAddCalls_ = 0;
};

} // namespace

// Pure-attention `seq_rm + seq_add` rejection MUST surface as
// `FailedKvIntact` (not `FailedKvWiped`) so the caller can roll back
// `[preRequestCursor, currentCursor)` on live KV instead of resetting
// to zero. Regression coverage for the single-prompt hardening in
// `TextLlmContext::compactThinkSpan` / `MtmdLlmContext::compactThinkSpan`
// where the previous catch handler reset positional bookkeeping to
// zero on this failure, leaving driver metadata and live KV out of
// sync for the next request on the same driver.
TEST(
    ReasoningBlockCompactorFailureStats,
    PureAttentionSeqRmRejectionReportsFailedKvIntact) {
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  // Pure-attention path: no recurrent snapshot needed. This is the
  // configuration that must produce `FailedKvIntact` on rejection.
  fx.compactor.setNeedsRecurrentSnapshot(false);

  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/20);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  RejectingSliderOps rejecting;
  fx.compactor.setContextSliderOpsForTesting(&rejecting);
  // `ctx` is passed through untouched by the fake ops; safe to pass
  // nullptr because neither `memory` nor `nCtx` inspects it.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
  fx.compactor.setContextSliderOpsForTesting(nullptr);

  EXPECT_EQ(
      outcome.kind, ReasoningBlockCompactor::Outcome::Kind::FailedKvIntact);
  EXPECT_FALSE(outcome.failureMessage.empty())
      << "failureMessage must be populated so caller can rethrow with "
         "matching diagnostic context";
  EXPECT_EQ(rejecting.seqRmCalls(), 1)
      << "compactor must attempt the pure-attention primitive exactly once";
  EXPECT_EQ(rejecting.seqAddCalls(), 0)
      << "seq_rm rejection must short-circuit before seq_add fires — "
         "otherwise the `FailedKvIntact` invariant (live KV unchanged) is "
         "violated";
  EXPECT_EQ(fx.compactor.blockDiscards(), 0)
      << "failed drops must not bump the runtime discard counter";

  // The `ResetGuard` still clears per-inference bookkeeping on the
  // failure return so a follow-up compact() on the same instance
  // starts clean.
  EXPECT_FALSE(fx.compactor.hasOpenSpan());
}

// ============================================================================
// tools_compact × remove_thinking_from_context — shared post-generation seam
// ============================================================================
//
// `TextLlmContext::onGenerationFinished` runs the two post-generation
// policies back-to-back: `onGenerationCompletePolicy` (tools_compact
// tail trim) fires first, then `compactThinkSpan()` (the reasoning
// compactor). Prior to this PR, no unit or integration test enabled
// both features at the same time — `reasoning.test.js` never sets
// `tools_compact` and `tools-compact.test.js` never sets
// `remove_thinking_from_context`. Pin the two invariants that connect
// them on the shared code path so a future change to either policy
// cannot silently break the other:
//
//   1. If a tail-eraser (today: tools_compact) has shrunk `nPast_`
//      below the recorded close-span end, `compact()` MUST behave
//      per-path:
//        a. Whole span already past the live cursor (`start >= pos`):
//           NoOp — nothing resident to remove.
//        b. Partial span still resident (`start < pos < end`):
//           * Pure-attention: honor the default-on strict-cleanup
//             contract by dropping the resident remainder via a
//             clamped `[start, pos)` `seq_rm + seq_add`; reports
//             `CompactedAttention`.
//           * Recurrent / hybrid: NoOp — replay is anchored at a
//             captured post-reasoning tail we can no longer reconcile
//             against a shorter live cache without the driver's
//             pre-request rollback anchor.
//      The current Qwen3-only tools_compact caller is not expected to
//      overshoot `</think>` (its trim is sized against the trailing
//      tool region only); these guards are the defence-in-depth path
//      if a future tail-eraser ever legitimately trims past the close
//      marker.
//   2. On a successful pure-attention drop, `compact()` MUST notify an
//      enabled `ToolsCompactController` via `onSlide` so the tools
//      anchor tracks the shifted tail. Skipping this would leave the
//      anchor pointing past the actual tool region on the next slide,
//      breaking `clampDiscard`.

TEST(
    ReasoningBlockCompactorToolsCompactInteraction,
    NoOpWhenWholeSpanTrimmedPastLivePos) {
  // Whole recorded reasoning span sits past the live cursor: `start`
  // and `end` are both above `pos`, so nothing from the span remains
  // resident. This models a tail-eraser that reset `pos` to a point
  // before the reasoning span (the shape produced by tools_compact
  // trimming the entire assistant tail back to `nPastBeforeTools_`).
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(false); // pure-attention

  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/25);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  AcceptingSliderOps accepting;
  fx.compactor.setContextSliderOpsForTesting(&accepting);
  // `pos = 10 <= start = 15`: reasoning span already gone from cache;
  // NoOp is the correct — not a leak — outcome.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/10, "[Test]");
  fx.compactor.setContextSliderOpsForTesting(nullptr);

  EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
  EXPECT_EQ(accepting.seqRmCalls(), 0)
      << "when the span is already trimmed away, no KV primitive must "
         "fire";
  EXPECT_EQ(accepting.seqAddCalls(), 0);
  EXPECT_EQ(fx.compactor.blockDiscards(), 0)
      << "NoOp on whole-span-trimmed must not be counted as a discard";

  // `ResetGuard` still clears per-inference state on the NoOp return.
  EXPECT_FALSE(fx.compactor.hasOpenSpan());
}

TEST(
    ReasoningBlockCompactorToolsCompactInteraction,
    PartialResidentSpanCompactsOnPureAttention) {
  // `start < pos < end`: the tail-eraser stopped inside the reasoning
  // span, so `[start, pos)` is still resident. Under the default-on
  // `remove_thinking_from_context` contract the compactor must not
  // silently leak reasoning tokens — the pure-attention path clamps
  // the effective end to `pos` and drops the resident remainder.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(false); // pure-attention

  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/25);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  AcceptingSliderOps accepting;
  fx.compactor.setContextSliderOpsForTesting(&accepting);
  // `pos = 20`, recordedEnd = 25 → effectiveEnd clamped to 20, so the
  // compactor drops `[15, 20)` — 5 tokens.
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/20, "[Test]");
  fx.compactor.setContextSliderOpsForTesting(nullptr);

  EXPECT_EQ(
      outcome.kind, ReasoningBlockCompactor::Outcome::Kind::CompactedAttention);
  EXPECT_EQ(outcome.newPos, 15)
      << "after clamped seq_rm, newPos falls to the reasoning span start";
  EXPECT_EQ(outcome.discarded, 5)
      << "discard length must equal the resident remainder `pos - start`";
  EXPECT_EQ(outcome.keptPrefixEnd, 15);
  EXPECT_EQ(accepting.seqRmCalls(), 1)
      << "clamped partial cleanup must issue the pure-attention seq_rm";
  EXPECT_EQ(accepting.seqAddCalls(), 1)
      << "successful seq_rm must be followed by its paired seq_add";
  EXPECT_EQ(fx.compactor.blockDiscards(), 1)
      << "clamped partial drop is still a real discard and must be counted";

  EXPECT_FALSE(fx.compactor.hasOpenSpan());
}

TEST(
    ReasoningBlockCompactorToolsCompactInteraction,
    NoOpWhenPartialResidentSpanOnRecurrentPath) {
  // Same partial-resident shape as above but on the recurrent /
  // hybrid path: replay is anchored at a captured post-reasoning tail
  // that no longer matches the shorter live cache, and there is no
  // safe way to reconcile without the driver's pre-request rollback
  // anchor. NoOp here — the driver's own recovery path (context
  // rollback or full reset on the next request) handles cleanup.
  CompactorFixture fx;
  fx.compactor.setRemoveThinkingFromContext(true);
  fx.compactor.setReasoningEnabled(true);
  fx.compactor.setNeedsRecurrentSnapshot(true); // recurrent / hybrid
  // `setOpenSpan` refuses the recurrent+no-boundary combination, so a
  // sentinel boundary snapshot is required for the span to be seeded
  // at all. `nPast=10` is arbitrary — the recurrent NoOp bail returns
  // before consulting the boundary payload.
  fx.rollback.seedReasoningBoundaryForTesting(/*nPast=*/10);
  ASSERT_TRUE(fx.rollback.hasReasoningBoundary());

  fx.compactor.setOpenSpan(/*start=*/15);
  fx.compactor.requestCloseCapture();
  fx.compactor.onCloseCommitted(/*pos=*/25);
  ASSERT_TRUE(fx.compactor.hasCapturedCloseSpanForTesting());

  AcceptingSliderOps accepting;
  fx.compactor.setContextSliderOpsForTesting(&accepting);
  const auto outcome = fx.compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/20, "[Test]");
  fx.compactor.setContextSliderOpsForTesting(nullptr);

  EXPECT_EQ(outcome.kind, ReasoningBlockCompactor::Outcome::Kind::NoOp);
  EXPECT_EQ(accepting.seqRmCalls(), 0)
      << "recurrent partial-resident NoOp must not touch any KV primitive";
  EXPECT_EQ(accepting.seqAddCalls(), 0);
  EXPECT_EQ(fx.compactor.blockDiscards(), 0)
      << "recurrent NoOp bail must not be counted as a successful discard";

  EXPECT_FALSE(fx.compactor.hasOpenSpan());
}

TEST(
    ReasoningBlockCompactorToolsCompactInteraction,
    SuccessfulPureAttentionDropNotifiesEnabledToolsController) {
  // Enable both features and drive a successful pure-attention
  // compaction. Assert that the compactor threads its discard through
  // `ToolsCompactController::onSlide` so the tools anchor shifts by
  // the same amount the tail shrank.
  ReasoningRollbackState rollback;
  ToolsCompactController tools{ToolsCompactProfile{}};
  ASSERT_TRUE(tools.enabled());

  // Seed the tools controller with an anchor via the normal lifecycle:
  //   - `onTokenize` captures the conversation-only token count,
  //   - `onEvalComplete` derives `nPastBeforeTools_` from the delta.
  //
  // Concrete numbers: total-with-tools=100, without-tools=80 =>
  // `nConversationOnlyTokens_ = 80`. After
  // `onEvalComplete(nPast=100, totalTokensEvaled=100)` the anchor
  // lands at `100 - (100 - 80) = 80`.
  constexpr size_t kWithTools = 100;
  constexpr size_t kWithoutTools = 80;
  constexpr llama_pos kNPastAfterEval = 100;
  tools.onTokenize(kWithTools, kWithoutTools);
  tools.onEvalComplete(kNPastAfterEval, /*totalTokensEvaled=*/kNPastAfterEval);
  ASSERT_EQ(tools.anchor(), 80);

  ReasoningBlockCompactor compactor{rollback, tools};
  compactor.setRemoveThinkingFromContext(true);
  compactor.setReasoningEnabled(true);
  compactor.setNeedsRecurrentSnapshot(false); // pure-attention

  // Reasoning close span at `[15, 20)`, live pos at 25 — 5 tokens will
  // be dropped by the successful compact.
  compactor.setOpenSpan(/*start=*/15);
  compactor.requestCloseCapture();
  compactor.onCloseCommitted(/*pos=*/20);

  AcceptingSliderOps accepting;
  compactor.setContextSliderOpsForTesting(&accepting);
  const auto outcome = compactor.compact(
      /*ctx=*/nullptr, /*seqId=*/0, /*pos=*/25, "[Test]");
  compactor.setContextSliderOpsForTesting(nullptr);

  EXPECT_EQ(
      outcome.kind, ReasoningBlockCompactor::Outcome::Kind::CompactedAttention);
  EXPECT_EQ(outcome.newPos, 20);
  EXPECT_EQ(outcome.discarded, 5);
  EXPECT_EQ(outcome.keptPrefixEnd, 15)
      << "after seq_rm + seq_add, the protected prefix ends at the span start";
  EXPECT_EQ(accepting.seqRmCalls(), 1);
  EXPECT_EQ(accepting.seqAddCalls(), 1)
      << "successful seq_rm must be followed by the paired seq_add";
  EXPECT_EQ(compactor.blockDiscards(), 1);

  // The whole point of this test: tools_compact must observe the drop.
  // Anchor should shift from 80 to 75 via `onSlide(5, /*first=*/15)`.
  // Without the `tools_.onSlide` call inside `compact()`, the anchor
  // would stay at 80 and the next `clampDiscard` would allow a slide
  // that eats into the tool region.
  EXPECT_EQ(tools.anchor(), 75)
      << "compactor must forward the discard through tools_.onSlide so the "
         "anchor tracks the shifted tail";
}

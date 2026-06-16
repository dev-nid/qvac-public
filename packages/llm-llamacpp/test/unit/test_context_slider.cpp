// Owns ContextSlider orchestration via trySlidePrefill/trySlideGeneration.
// Controller primitive behavior is intentionally covered in
// test_tools_compact_controller.cpp.

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "model-interface/ContextSlider.hpp"
#include "model-interface/ToolsCompactController.hpp"

namespace {
constexpr llama_seq_id kSeqId = 7;

struct SeqRmCall {
  llama_seq_id seqId = 0;
  llama_pos startPos = 0;
  llama_pos endPos = 0;
};

bool operator==(const SeqRmCall& lhs, const SeqRmCall& rhs) {
  return lhs.seqId == rhs.seqId && lhs.startPos == rhs.startPos &&
         lhs.endPos == rhs.endPos;
}

struct SeqAddCall {
  llama_seq_id seqId = 0;
  llama_pos startPos = 0;
  llama_pos endPos = 0;
  llama_pos delta = 0;
};

class FakeLlamaContextOps final : public IContextSliderOps {
public:
  explicit FakeLlamaContextOps(llama_pos ctxSize) : ctxSize_(ctxSize) {}

  llama_pos nCtx(llama_context*) const override { return ctxSize_; }

  ContextSliderMemoryHandle memory(llama_context*) const override {
    ++memoryCalls_;
    return fakeMemory_;
  }

  bool seqRm(
      ContextSliderMemoryHandle mem, llama_seq_id seqId, llama_pos startPos,
      llama_pos endPos) const override {
    EXPECT_EQ(mem, fakeMemory_);
    seqRmCalls_.push_back({seqId, startPos, endPos});

    if (seqRmFailure_ && seqRmFailure_->seqId == seqId &&
        seqRmFailure_->startPos == startPos &&
        seqRmFailure_->endPos == endPos) {
      return false;
    }

    return true;
  }

  void seqAdd(
      ContextSliderMemoryHandle mem, llama_seq_id seqId, llama_pos startPos,
      llama_pos endPos, llama_pos delta) const override {
    EXPECT_EQ(mem, fakeMemory_);
    seqAddCalls_.push_back({seqId, startPos, endPos, delta});
  }

  int memoryCalls() const { return memoryCalls_; }
  const std::vector<SeqRmCall>& seqRmCalls() const { return seqRmCalls_; }
  const std::vector<SeqAddCall>& seqAddCalls() const { return seqAddCalls_; }
  void failSeqRmFor(SeqRmCall call) { seqRmFailure_ = call; }

private:
  llama_pos ctxSize_;
  ContextSliderMemoryHandle fakeMemory_ =
      reinterpret_cast<ContextSliderMemoryHandle>(static_cast<uintptr_t>(0x1));
  mutable int memoryCalls_ = 0;
  mutable std::vector<SeqRmCall> seqRmCalls_;
  mutable std::vector<SeqAddCall> seqAddCalls_;
  std::optional<SeqRmCall> seqRmFailure_;
};
} // namespace

class ContextSliderTest : public ::testing::Test {};

TEST_F(ContextSliderTest, PrefillSlideScenario_EnoughRoom) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/500);

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/100,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/50,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::NotNeeded);
  EXPECT_EQ(outcome.newNPast, 100);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(ops.memoryCalls(), 0);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, PrefillSlidInvokesLlamaOpsWithExpectedRanges) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/400);

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/300,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/180,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Slid);
  EXPECT_EQ(outcome.newNPast, 200);
  EXPECT_EQ(outcome.discarded, 100);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 150);

  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 150);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 300);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -100);
}

TEST_F(ContextSliderTest, PrefillSlideReturnsMemoryFailureWhenSeqRmFails) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/400);
  ops.failSeqRmFor({kSeqId, 50, 150});

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/300,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/180,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::MemoryOperationFailed);
  EXPECT_EQ(outcome.newNPast, 300);
  EXPECT_EQ(outcome.discarded, 0);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0], (SeqRmCall{kSeqId, 50, 150}));
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

// Batch mode partitions the KV pool into per-slot caps (ctx / n_parallel)
// that are far smaller than the whole-context size. A cached prompt can fit
// the full context yet overflow its slot; the slide must trigger against the
// per-sequence cap so n_discarded can free room before the scheduler rejects
// the prompt. Regression for PR #2327 review r3344885390.
TEST_F(ContextSliderTest, PrefillSlidesAgainstPerSeqCapBelowFullCtx) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/8192);

  // nPast + append = 2100: over the per-seq cap (2048) but well under the
  // full context (8192). Sliding against the full ctx would do nothing.
  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/1900,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/200,
      /*nDiscarded=*/512,
      controller,
      ops,
      /*effectiveCtx=*/2048);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Slid);
  EXPECT_EQ(outcome.newNPast, 1388);
  EXPECT_EQ(outcome.discarded, 512);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 562);

  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 562);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 1900);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -512);
}

TEST_F(ContextSliderTest, PrefillFullWipeInvokesSeqRmOnly) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/300);

  controller.onTokenize(120, 50);
  controller.onEvalComplete(120, 120);
  EXPECT_EQ(controller.anchor(), 50);

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/120,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/200,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::FullWipe);
  EXPECT_EQ(outcome.newNPast, 50);
  EXPECT_EQ(outcome.discarded, 70);
  EXPECT_EQ(controller.anchor(), -1);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 120);
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, PrefillFullWipePreservesTailWhenExactWipeFails) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/300);

  controller.onTokenize(120, 50);
  controller.onEvalComplete(120, 120);
  ops.failSeqRmFor({kSeqId, 50, 120});

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/120,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/200,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::FullWipe);
  EXPECT_EQ(outcome.newNPast, 51);
  EXPECT_EQ(outcome.discarded, 69);
  EXPECT_EQ(controller.anchor(), -1);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 2u);
  EXPECT_EQ(ops.seqRmCalls()[0], (SeqRmCall{kSeqId, 50, 120}));
  EXPECT_EQ(ops.seqRmCalls()[1], (SeqRmCall{kSeqId, 50, 119}));

  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 119);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 120);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -69);
}

TEST_F(ContextSliderTest, PrefillFullWipeWhenPartialSlideCannotFit) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/512);

  controller.onTokenize(474, 25);
  controller.onEvalComplete(474, 474);
  ops.failSeqRmFor({kSeqId, 25, 474});

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/474,
      /*firstMsgTokens=*/25,
      /*nTokensToAppend=*/308,
      /*nDiscarded=*/512,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::FullWipe);
  EXPECT_EQ(outcome.newNPast, 26);
  EXPECT_EQ(outcome.discarded, 448);
  EXPECT_EQ(controller.anchor(), -1);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 2u);
  EXPECT_EQ(ops.seqRmCalls()[0], (SeqRmCall{kSeqId, 25, 474}));
  EXPECT_EQ(ops.seqRmCalls()[1], (SeqRmCall{kSeqId, 25, 473}));

  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 473);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 474);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -448);
}

TEST_F(ContextSliderTest, PrefillFullWipeRespectsDiscardBudget) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/512);

  controller.onTokenize(474, 25);
  controller.onEvalComplete(474, 474);

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/474,
      /*firstMsgTokens=*/25,
      /*nTokensToAppend=*/308,
      /*nDiscarded=*/256,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Overflow);
  EXPECT_EQ(outcome.newNPast, 474);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(controller.anchor(), 25);
  EXPECT_EQ(ops.memoryCalls(), 0);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, PrefillSlideScenario_Overflow) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/100);

  ContextSlideOutcome outcome = trySlidePrefill(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/75,
      /*firstMsgTokens=*/50,
      /*nTokensToAppend=*/200,
      /*nDiscarded=*/100,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Overflow);
  EXPECT_EQ(outcome.newNPast, 75);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(ops.memoryCalls(), 0);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, GenerationSlideScenario_EnoughRoom) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/500);

  ContextSlideOutcome outcome = trySlideGeneration(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/499,
      /*firstMsgTokens=*/50,
      /*nDiscarded=*/120,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::NotNeeded);
  EXPECT_EQ(outcome.newNPast, 499);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(ops.memoryCalls(), 0);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, GenerationSlidInvokesLlamaOpsWithExpectedRanges) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/400);

  ContextSlideOutcome outcome = trySlideGeneration(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/400,
      /*firstMsgTokens=*/50,
      /*nDiscarded=*/120,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Slid);
  EXPECT_EQ(outcome.newNPast, 280);
  EXPECT_EQ(outcome.discarded, 120);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 170);

  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 170);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 400);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -120);
}

TEST_F(ContextSliderTest, GenerationSlideScenario_NoDiscardAllowed) {
  ToolsCompactController controller(std::nullopt);
  FakeLlamaContextOps ops(/*ctxSize=*/500);

  ContextSlideOutcome outcome = trySlideGeneration(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/500,
      /*firstMsgTokens=*/50,
      /*nDiscarded=*/0,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::NotNeeded);
  EXPECT_EQ(outcome.newNPast, 500);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(ops.memoryCalls(), 0);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, GenerationToolsCompactClampsDiscardToAnchorWindow) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/140);
  constexpr llama_pos firstMsgTokens = 50;

  controller.onTokenize(/*tokensWithTools=*/140, /*tokensWithoutTools=*/80);
  controller.onEvalComplete(/*nPast=*/140, /*totalTokensEvaled=*/140);
  ASSERT_EQ(controller.anchor(), 80);

  ContextSlideOutcome outcome = trySlideGeneration(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/140,
      firstMsgTokens,
      /*nDiscarded=*/120,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Slid);
  EXPECT_EQ(outcome.newNPast, 110);
  EXPECT_EQ(outcome.discarded, 30);
  EXPECT_EQ(controller.anchor(), firstMsgTokens);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 80);
  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 80);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 140);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -30);
}

TEST_F(
    ContextSliderTest,
    GenerationDegenerateBoundaryResetsThenSlidesFromFirstMessage) {
  ToolsCompactController controller(ToolsCompactProfile{});
  FakeLlamaContextOps ops(/*ctxSize=*/120);
  constexpr llama_pos firstMsgTokens = 50;

  controller.onTokenize(/*tokensWithTools=*/120, /*tokensWithoutTools=*/50);
  controller.onEvalComplete(/*nPast=*/120, /*totalTokensEvaled=*/120);
  ASSERT_EQ(controller.anchor(), firstMsgTokens);
  ASSERT_TRUE(controller.degenerateBoundary(firstMsgTokens));

  ContextSlideOutcome outcome = trySlideGeneration(
      /*lctx=*/nullptr,
      kSeqId,
      /*nPast=*/120,
      firstMsgTokens,
      /*nDiscarded=*/40,
      controller,
      ops);

  EXPECT_EQ(outcome.kind, ContextSlideOutcome::Kind::Slid);
  EXPECT_EQ(outcome.newNPast, 80);
  EXPECT_EQ(outcome.discarded, 40);
  EXPECT_EQ(controller.anchor(), firstMsgTokens);

  ASSERT_EQ(ops.memoryCalls(), 1);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 50);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 90);
  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 90);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 120);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -40);
}

// ---------------------------------------------------------------------------
// compactKvRange — used by `TextLlmContext::compactThinkSpans` to drop a
// model's reasoning block from the KV cache without involving the regular
// slide policy logic.
// ---------------------------------------------------------------------------

TEST_F(ContextSliderTest, CompactKvRange_HappyPath_RemovesRangeAndShiftsTail) {
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  // Cache layout:  [user prompt 100][reasoning 50][answer 30]
  // We want to drop the reasoning block at [100, 150).
  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/100,
      /*endPos=*/150,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::Compacted);
  EXPECT_EQ(outcome.discarded, 50);
  EXPECT_EQ(outcome.newNPast, 130);

  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_EQ(ops.seqRmCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqRmCalls()[0].startPos, 100);
  EXPECT_EQ(ops.seqRmCalls()[0].endPos, 150);

  // The surviving tail `[150, 180)` should shift down by 50 to occupy
  // `[100, 130)`.
  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].seqId, kSeqId);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 150);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 180);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -50);
}

TEST_F(ContextSliderTest, CompactKvRange_EmptyRange_IsNoOp) {
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/100,
      /*endPos=*/100,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::NoOp);
  EXPECT_EQ(outcome.discarded, 0);
  EXPECT_EQ(outcome.newNPast, 180);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, CompactKvRange_InvertedRange_IsNoOp) {
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/150,
      /*endPos=*/100,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::NoOp);
  EXPECT_EQ(outcome.newNPast, 180);
  EXPECT_TRUE(ops.seqRmCalls().empty());
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, CompactKvRange_EndPastNPast_IsNoOp) {
  // Defensive: end > nPast means the recorded span is stale (e.g. a
  // slide already discarded some of those tokens). Refuse to compact
  // rather than corrupt the cache.
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/100,
      /*endPos=*/200,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::NoOp);
  EXPECT_EQ(outcome.newNPast, 180);
  EXPECT_TRUE(ops.seqRmCalls().empty());
}

TEST_F(ContextSliderTest, CompactKvRange_NegativeStart_IsNoOp) {
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/-1,
      /*endPos=*/50,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::NoOp);
  EXPECT_EQ(outcome.newNPast, 180);
  EXPECT_TRUE(ops.seqRmCalls().empty());
}

TEST_F(ContextSliderTest, CompactKvRange_SeqRmFailure_ReportsAndSkipsShift) {
  FakeLlamaContextOps ops(/*ctxSize=*/1024);
  ops.failSeqRmFor({.seqId = kSeqId, .startPos = 100, .endPos = 150});

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/100,
      /*endPos=*/150,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::MemoryOperationFailed);
  // Caller must not advance bookkeeping on a failed compaction.
  EXPECT_EQ(outcome.newNPast, 180);
  EXPECT_EQ(outcome.discarded, 0);
  // seqRm was attempted but failed; seqAdd must NOT run, otherwise the
  // cache would be shifted without the corresponding window removed.
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  EXPECT_TRUE(ops.seqAddCalls().empty());
}

TEST_F(ContextSliderTest, CompactKvRange_TailExactlyAtEnd_NoShiftNeeded) {
  // Range covers everything from `startPos` to `nPast`. The shift is a
  // no-op range `[end, end)` for `seqAdd`, but we still expect it to
  // be invoked (the slider does not branch on empty tails — `seqAdd`
  // with start==end is a cheap no-op for the underlying llama API).
  FakeLlamaContextOps ops(/*ctxSize=*/1024);

  const auto outcome = compactKvRange(
      /*lctx=*/nullptr,
      kSeqId,
      /*startPos=*/100,
      /*endPos=*/180,
      /*nPast=*/180,
      ops);

  EXPECT_EQ(outcome.kind, CompactRangeOutcome::Kind::Compacted);
  EXPECT_EQ(outcome.discarded, 80);
  EXPECT_EQ(outcome.newNPast, 100);
  ASSERT_EQ(ops.seqRmCalls().size(), 1u);
  ASSERT_EQ(ops.seqAddCalls().size(), 1u);
  EXPECT_EQ(ops.seqAddCalls()[0].startPos, 180);
  EXPECT_EQ(ops.seqAddCalls()[0].endPos, 180);
  EXPECT_EQ(ops.seqAddCalls()[0].delta, -80);
}

// ---------------------------------------------------------------------------
// runReasoningCompaction — thinking-block span orchestration. This is the
// part of `TextLlmContext::compactThinkSpans` that used to be a hand-rolled
// loop and is now extracted so the per-span semantics (reverse-walk,
// firstMsgTokens shrink, onSlide args, skip rules) can be unit-tested
// without a real llama context. Tests below pass deterministic fake
// `compactOp` / `onSlide` callables and assert on the result state +
// recorded calls.
// ---------------------------------------------------------------------------

namespace {

struct CompactOpCall {
  llama_pos start;
  llama_pos end;
  llama_pos nPast;
};

struct SlideCall {
  int discarded;
  llama_pos start;
};

// Fake compactor: always succeeds and shifts nPast by `end - start`.
// Records each call so tests can assert the reverse-walk order.
struct AlwaysCompactingFake {
  std::vector<CompactOpCall> calls;
  CompactRangeOutcome operator()(
      llama_pos start, llama_pos end, llama_pos nPast) {
    calls.push_back({start, end, nPast});
    const llama_pos discarded = end - start;
    return {CompactRangeOutcome::Kind::Compacted, nPast - discarded, discarded};
  }
};

} // namespace

TEST_F(ContextSliderTest, ReasoningCompaction_SingleSpan_UpdatesStateAndNotifies) {
  AlwaysCompactingFake compactor;
  std::vector<SlideCall> slides;
  // Span [120, 200) inside the active tail (firstMsgTokens=80, nPast=240).
  const std::vector<std::pair<llama_pos, llama_pos>> spans{{120, 200}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/240,
      /*firstMsgTokens=*/80,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  EXPECT_EQ(result.newNPast, 160);             // 240 - 80
  EXPECT_EQ(result.newFirstMsgTokens, 80);     // unchanged (span past prefix)
  EXPECT_EQ(result.discardsApplied, 1);
  ASSERT_EQ(result.entries.size(), 1u);
  EXPECT_EQ(result.entries[0].status,
            ReasoningCompactionEntry::Status::Compacted);
  EXPECT_EQ(result.entries[0].discarded, 80);
  ASSERT_EQ(slides.size(), 1u);
  EXPECT_EQ(slides[0].discarded, 80);
  EXPECT_EQ(slides[0].start, 120);
}

TEST_F(ContextSliderTest, ReasoningCompaction_SpanBeforeFirstMsg_ShrinksPrefix) {
  AlwaysCompactingFake compactor;
  std::vector<SlideCall> slides;
  // Template-forced open span at [50, 130): start < firstMsgTokens (80).
  const std::vector<std::pair<llama_pos, llama_pos>> spans{{50, 130}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/200,
      /*firstMsgTokens=*/80,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  EXPECT_EQ(result.newNPast, 120);             // 200 - 80
  EXPECT_EQ(result.newFirstMsgTokens, 50);     // clamped down to span start
  EXPECT_EQ(result.discardsApplied, 1);
  ASSERT_EQ(slides.size(), 1u);
  EXPECT_EQ(slides[0].discarded, 80);
  EXPECT_EQ(slides[0].start, 50);
}

TEST_F(ContextSliderTest, ReasoningCompaction_MultipleSpans_ReverseDropOrder) {
  AlwaysCompactingFake compactor;
  std::vector<SlideCall> slides;
  // Two spans recorded in forward order: [100, 130), [180, 220). Walk must
  // process the LATER span first so the earlier span's positions stay
  // valid for its own seqRm call.
  const std::vector<std::pair<llama_pos, llama_pos>> spans{{100, 130}, {180, 220}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/250,
      /*firstMsgTokens=*/50,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  // First call should be the later span [180, 220) against nPast=250.
  ASSERT_EQ(compactor.calls.size(), 2u);
  EXPECT_EQ(compactor.calls[0].start, 180);
  EXPECT_EQ(compactor.calls[0].end, 220);
  EXPECT_EQ(compactor.calls[0].nPast, 250);
  // Second call should be the earlier span [100, 130) against the updated
  // nPast (250 - 40 = 210).
  EXPECT_EQ(compactor.calls[1].start, 100);
  EXPECT_EQ(compactor.calls[1].end, 130);
  EXPECT_EQ(compactor.calls[1].nPast, 210);

  EXPECT_EQ(result.newNPast, 180);             // 250 - 40 - 30
  EXPECT_EQ(result.discardsApplied, 2);
  ASSERT_EQ(slides.size(), 2u);
  EXPECT_EQ(slides[0].start, 180);
  EXPECT_EQ(slides[1].start, 100);
}

TEST_F(ContextSliderTest, ReasoningCompaction_OpenSpan_Skipped) {
  AlwaysCompactingFake compactor;
  std::vector<SlideCall> slides;
  // Span with end < 0 (close was never captured) plus a real one.
  const std::vector<std::pair<llama_pos, llama_pos>> spans{
      {120, static_cast<llama_pos>(-1)}, {180, 220}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/250,
      /*firstMsgTokens=*/50,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  // Only the closed span hit the compactor.
  ASSERT_EQ(compactor.calls.size(), 1u);
  EXPECT_EQ(compactor.calls[0].start, 180);
  EXPECT_EQ(result.discardsApplied, 1);
  ASSERT_EQ(slides.size(), 1u);
  // Reverse walk: the open span is the second entry in `result.entries`
  // (walked first, recorded second).
  ASSERT_EQ(result.entries.size(), 2u);
  EXPECT_EQ(result.entries[0].status,
            ReasoningCompactionEntry::Status::Compacted);
  EXPECT_EQ(result.entries[1].status,
            ReasoningCompactionEntry::Status::Skipped);
}

TEST_F(ContextSliderTest, ReasoningCompaction_DegenerateSpan_Skipped) {
  AlwaysCompactingFake compactor;
  std::vector<SlideCall> slides;
  // end == start (empty) and end < start (inverted) — both must skip
  // without invoking compactOp.
  const std::vector<std::pair<llama_pos, llama_pos>> spans{
      {100, 100}, {150, 130}, {180, 220}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/250,
      /*firstMsgTokens=*/50,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  ASSERT_EQ(compactor.calls.size(), 1u);
  EXPECT_EQ(compactor.calls[0].start, 180);
  EXPECT_EQ(result.discardsApplied, 1);
  ASSERT_EQ(result.entries.size(), 3u);
  // Reverse walk: real span processed first → entries[0]. Two skipped
  // degenerate spans follow.
  EXPECT_EQ(result.entries[0].status,
            ReasoningCompactionEntry::Status::Compacted);
  EXPECT_EQ(result.entries[1].status,
            ReasoningCompactionEntry::Status::Skipped);
  EXPECT_EQ(result.entries[2].status,
            ReasoningCompactionEntry::Status::Skipped);
}

TEST_F(ContextSliderTest, ReasoningCompaction_MemoryOperationFailed_NoStateChange) {
  // Fake that fails the FIRST call (the later span in reverse walk).
  struct FailingFake {
    std::vector<CompactOpCall> calls;
    CompactRangeOutcome operator()(
        llama_pos start, llama_pos end, llama_pos nPast) {
      calls.push_back({start, end, nPast});
      if (calls.size() == 1) {
        return {CompactRangeOutcome::Kind::MemoryOperationFailed, nPast, 0};
      }
      const llama_pos discarded = end - start;
      return {
          CompactRangeOutcome::Kind::Compacted, nPast - discarded, discarded};
    }
  } compactor;
  std::vector<SlideCall> slides;
  const std::vector<std::pair<llama_pos, llama_pos>> spans{{100, 130}, {180, 220}};

  const auto result = runReasoningCompaction(
      spans,
      /*nPast=*/250,
      /*firstMsgTokens=*/50,
      std::ref(compactor),
      [&](int d, llama_pos s) { slides.push_back({d, s}); });

  // First (reverse) span failed: nPast unchanged for that step. Second
  // (earlier) span proceeds normally against the still-untouched tail.
  EXPECT_EQ(result.newNPast, 220);             // 250 - 30 only
  EXPECT_EQ(result.discardsApplied, 1);
  ASSERT_EQ(result.entries.size(), 2u);
  EXPECT_EQ(result.entries[0].status,
            ReasoningCompactionEntry::Status::MemoryOperationFailed);
  EXPECT_EQ(result.entries[1].status,
            ReasoningCompactionEntry::Status::Compacted);
  // onSlide only fires for the successful compaction.
  ASSERT_EQ(slides.size(), 1u);
  EXPECT_EQ(slides[0].start, 100);
}

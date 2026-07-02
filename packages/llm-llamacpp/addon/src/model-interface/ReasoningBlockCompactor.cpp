#include "ReasoningBlockCompactor.hpp"

#include <cstddef>
#include <utility>

#include <common/common.h>
#include <inference-addon-cpp/Errors.hpp>
#include <llama.h>

#include "../addon/LlmErrors.hpp"
#include "../utils/LoggingMacros.hpp"
#include "../utils/ReasoningRollbackState.hpp"
#include "ContextSlider.hpp"
#include "ToolsCompactController.hpp"
#include "inference-addon-cpp/Logger.hpp"

using namespace qvac_lib_inference_addon_cpp::logger;

namespace qvac_lib_inference_addon_llama {

namespace {

// Best-effort sequence wipe used before throwing on the hybrid
// restore/replay failure path. On success live memory is empty for
// `seqId`, matching the caller's post-catch reset onto pos=0. Silent
// no-op when `ctx` is null (unit-test seam) or `llama_get_memory`
// returns null — the throw still fires below so the caller reacts
// appropriately, but we can't reason further about live state.
//
// `llama_memory_seq_rm(-1, -1)` is documented never to fail for a
// full-range delete (only partial ranges over recurrent memory can
// reject), but log a warning if it ever does so operators see the
// stale state rather than debugging silent cache-key drift later.
void clearSeqOnFailure(::llama_context* ctx, llama_seq_id seqId) noexcept {
  if (ctx == nullptr) {
    return;
  }
  auto* mem = llama_get_memory(ctx);
  if (mem == nullptr) {
    return;
  }
  const bool cleared = llama_memory_seq_rm(mem, seqId, -1, -1);
  if (!cleared) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "[ReasoningBlockCompactor] llama_memory_seq_rm(-1,-1) refused "
            "full-range wipe on seqId=%d before hard-fail throw; caller's "
            "post-catch reset may not match live memory\n",
            static_cast<int>(seqId)));
  }
}

} // namespace

ReasoningBlockCompactor::ReasoningBlockCompactor(
    utils::ReasoningRollbackState& rollback, ToolsCompactController& tools)
    : rollback_(rollback), tools_(tools) {}

void ReasoningBlockCompactor::setOpenSpan(llama_pos start) {
  // `start < 0` only for degenerate templates whose entire rendered
  // prompt is the forced-open suffix; drop the span and leave the
  // tokens in cache.
  if (!removeThinkingFromContext_ || !reasoningEnabled_ || start < 0) {
    return;
  }
  if (thinkSpan_.has_value()) {
    return;
  }
  thinkSpan_ = std::make_pair(start, static_cast<llama_pos>(-1));
}

void ReasoningBlockCompactor::recordCloseMarkerForReplay(llama_token id) {
  if (!removeThinkingFromContext_ || !reasoningEnabled_) {
    return;
  }
  if (!needsRecurrentSnapshot_ || !rollback_.hasReasoningBoundary()) {
    return;
  }
  rollback_.appendPostReasoningToken(id);
}

void ReasoningBlockCompactor::onCloseCommitted(llama_pos pos) {
  if (!pendingThinkCloseCapture_) {
    return;
  }
  pendingThinkCloseCapture_ = false;
  if (!removeThinkingFromContext_ || !thinkSpan_.has_value()) {
    return;
  }
  if (thinkSpan_->second < 0) {
    thinkSpan_->second = pos;
  }
  // Begin capturing post-reasoning tokens for replay against the
  // restored SSM. Only meaningful when a recurrent boundary snapshot
  // actually exists; pure-attention models leave the snapshot empty
  // and skip the replay path entirely.
  rollback_.startPostReasoningCapture(
      needsRecurrentSnapshot_ && rollback_.hasReasoningBoundary());
}

void ReasoningBlockCompactor::snapshotAtPrefillBoundary(
    ::llama_context* ctx, llama_seq_id seqId, llama_pos pos,
    const char* labelTag) {
  if (!needsRecurrentSnapshot_ || !removeThinkingFromContext_ ||
      !reasoningEnabled_) {
    return;
  }
  if (rollback_.hasReasoningBoundary()) {
    return; // already snapshotted this inference
  }
  if (!rollback_.captureReasoningBoundary(ctx, seqId, pos)) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "%s thinking-block compaction skipped: failed to snapshot "
            "sequence state at prefill boundary (pos=%d, seqId=%d)\n",
            labelTag,
            pos,
            seqId));
    ++thinkingCompactionFailed_;
  }
}

ReasoningBlockCompactor::Outcome ReasoningBlockCompactor::compact(
    ::llama_context* ctx, llama_seq_id seqId, llama_pos pos,
    const char* labelTag) {
  // RAII-style cleanup so every early return drops the per-inference
  // rollback buffers and span. The original sites in `TextLlmContext`
  // and `MtmdLlmContext` had identical guards; centralised here so
  // there is no drift.
  struct ResetGuard {
    ReasoningBlockCompactor* self;
    ~ResetGuard() {
      self->thinkSpan_.reset();
      self->rollback_.clearReasoningBoundary();
      self->rollback_.clearPostReasoning();
    }
  } guard{this};

  Outcome out;
  if (!removeThinkingFromContext_ || !thinkSpan_.has_value()) {
    return out;
  }
  const llama_pos start = thinkSpan_->first;
  const llama_pos end = thinkSpan_->second;
  out.spanStart = start;
  out.spanEnd = end;

  // Skip open (close never captured) or degenerate spans without
  // touching the cache. This is the single validation backstop for
  // all close-capture sites — none validate `end > start` themselves.
  if (end < 0 || end <= start) {
    return out;
  }
  // Defensive: if the tools-compact tail trim or any other tail-eraser
  // ran between span end and here, the recorded `end` may overshoot
  // the live cache. Bail rather than risk replaying tokens past the
  // committed tail.
  if (end > pos) {
    return out;
  }

  // Recurrent / hybrid models without a boundary snapshot cannot be
  // compacted safely: `seq_rm` over an interior range silently leaves
  // the SSM hidden state inconsistent, and there is no full-state
  // snapshot to roll back to. Skip compaction for this turn; the
  // capture-failure site already logged + bumped
  // `thinkingCompactionFailed_`.
  if (needsRecurrentSnapshot_ && !rollback_.hasReasoningBoundary()) {
    QLOG_IF(
        Priority::DEBUG,
        string_format(
            "%s thinking-block compaction skipped: recurrent / hybrid "
            "model has no boundary snapshot (start=%d, end=%d, pos=%d, "
            "seqId=%d)\n",
            labelTag,
            start,
            end,
            pos,
            seqId));
    return out;
  }

  // Pure-attention path: the SSM is uninvolved, so the seq_rm + seq_add
  // primitive is sufficient.
  if (!needsRecurrentSnapshot_) {
    const CompactRangeOutcome rangeOutcome =
        compactKvRange(ctx, seqId, start, end, pos);
    if (rangeOutcome.kind == CompactRangeOutcome::Kind::Compacted) {
      out.kind = Outcome::Kind::CompactedAttention;
      out.newPos = rangeOutcome.newNPast;
      out.discarded = rangeOutcome.discarded;
      // After `seq_rm + seq_add`, the tail shifts left into the slice
      // we just removed, so the protected-prefix end becomes `start`.
      out.keptPrefixEnd = start;
      if (tools_.enabled()) {
        tools_.onSlide(rangeOutcome.discarded, start);
      }
      ++thinkingBlockDiscards_;
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "%s thinking-block compaction: dropped %d tokens "
              "[%d, %d), newPos=%d\n",
              labelTag,
              rangeOutcome.discarded,
              start,
              end,
              out.newPos));
    } else if (
        rangeOutcome.kind == CompactRangeOutcome::Kind::MemoryOperationFailed) {
      out.kind = Outcome::Kind::FailedAttention;
      QLOG_IF(
          Priority::WARNING,
          string_format(
              "%s thinking-block compaction failed: seqRm rejected "
              "range [%d, %d) (pos=%d, seqId=%d)\n",
              labelTag,
              start,
              end,
              pos,
              seqId));
      ++thinkingCompactionFailed_;
    }
    return out;
  }

  // Recurrent / hybrid path. A `seq_rm` over a partial tail that
  // includes the final committed position is rejected by the
  // recurrent memory module, so we cannot use the pure-attention
  // primitive here. Instead:
  //   1. restore the FULL-state snapshot taken at the recurrent
  //      rollback boundary — this rebuilds both the attention KV and
  //      the recurrent state back to that point in one call; no
  //      `seq_rm` is needed.
  //   2. replay only the post-reasoning tokens through `llama_decode`
  //      starting at `snapshot.nPast`, so the new tokens occupy the
  //      cells immediately after the restored prefix.
  //
  // The kept prefix is `[0, snapshot.nPast)`. For forced-open
  // templates, that includes the opener residue documented by the
  // snapshot-at-end-of-prefill strategy.
  //
  // `pos - end` is the captured post-reasoning tail length (the live
  // cache holds tokens at positions `[end, pos)`). The replay buffer
  // additionally holds a seeded close marker at its head, which
  // `clipPostReasoningTokens` preserves regardless of the cap; passing
  // the captured-tail length here drops any captured tokens that the
  // tools-compact tail trim has since removed from the live cache,
  // without touching the structural prefix.
  const llama_pos snapshotPos = rollback_.reasoningBoundaryNPast();
  rollback_.clipPostReasoningTokens(static_cast<size_t>(pos - end));

  if (!rollback_.restoreReasoningBoundary(ctx, seqId)) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "%s thinking-block compaction failed: full-state restore "
            "underflowed (start=%d, end=%d, snapshotPos=%d, "
            "seqId=%d)\n",
            labelTag,
            start,
            end,
            snapshotPos,
            seqId));
    ++thinkingCompactionFailed_;
    // llama.cpp reports the load short-read but does not tell us
    // whether it left the sequence untouched or in a partially loaded
    // state. Either way it is unsafe to keep decoding into it: the
    // recurrent hidden state is not positionally indexed and cannot
    // be reasoned about after an aborted `state_seq_load_file`. Wipe
    // the sequence (attention KV cells + recurrent state) so the
    // caller's post-catch reset onto pos=0 matches live memory, then
    // fail hard so callers cannot save a cache whose header no longer
    // matches what is serialized.
    clearSeqOnFailure(ctx, seqId);
    throw qvac_errors::StatusError(
        errors::ADDON_ID,
        errors::toString(errors::FailedToDecode),
        string_format(
            "%s ReasoningBlockCompactor::compact: full-state restore "
            "underflowed on hybrid/recurrent compaction; sequence "
            "cleared (snapshotPos=%d, spanStart=%d, spanEnd=%d, "
            "seqId=%d)",
            labelTag,
            snapshotPos,
            start,
            end,
            seqId));
  }

  const size_t replayCount = rollback_.postReasoningTokenCount();
  if (!rollback_.replayPostReasoning(ctx, seqId)) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "%s thinking-block compaction failed: post-reasoning "
            "replay rejected (snapshotPos=%d, replayCount=%zu, "
            "seqId=%d)\n",
            labelTag,
            snapshotPos,
            replayCount,
            seqId));
    ++thinkingCompactionFailed_;
    // Restore succeeded, so live memory currently sits at
    // `snapshotPos`, but the replay decoded an unknown prefix of the
    // post-reasoning tokens before failing — the recurrent state has
    // partially advanced past `snapshotPos` with no way to observe
    // how far. Same coherence problem as restore failure; same fix.
    clearSeqOnFailure(ctx, seqId);
    throw qvac_errors::StatusError(
        errors::ADDON_ID,
        errors::toString(errors::FailedToDecode),
        string_format(
            "%s ReasoningBlockCompactor::compact: post-reasoning "
            "replay rejected on hybrid/recurrent compaction; sequence "
            "cleared (snapshotPos=%d, replayCount=%zu, seqId=%d)",
            labelTag,
            snapshotPos,
            replayCount,
            seqId));
  }

  const llama_pos newPos = snapshotPos + static_cast<llama_pos>(replayCount);
  out.kind = Outcome::Kind::CompactedRecurrent;
  out.newPos = newPos;
  out.discarded = pos - newPos;
  out.keptPrefixEnd = snapshotPos;
  out.replayedTokens = replayCount;
  if (tools_.enabled()) {
    tools_.onSlide(out.discarded, snapshotPos);
  }
  ++thinkingBlockDiscards_;
  QLOG_IF(
      Priority::DEBUG,
      string_format(
          "%s thinking-block compaction (recurrent): dropped %d tokens "
          "(span [%d, %d), kept [0, %d)), replayed %zu post-reasoning "
          "tokens, newPos=%d\n",
          labelTag,
          out.discarded,
          start,
          end,
          snapshotPos,
          replayCount,
          newPos));
  return out;
}

} // namespace qvac_lib_inference_addon_llama

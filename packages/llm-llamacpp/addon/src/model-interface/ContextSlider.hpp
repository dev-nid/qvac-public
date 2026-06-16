#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <llama.h>

class ToolsCompactController;

using ContextSliderMemoryHandle =
    decltype(llama_get_memory(static_cast<llama_context*>(nullptr)));

/// Small indirection layer around llama context/memory operations.
///
/// This makes ContextSlider testable without requiring a real llama_context.
struct IContextSliderOps {
  virtual ~IContextSliderOps() = default;
  virtual llama_pos nCtx(llama_context* lctx) const = 0;
  virtual ContextSliderMemoryHandle memory(llama_context* lctx) const = 0;
  virtual bool seqRm(
      ContextSliderMemoryHandle mem, llama_seq_id seqId, llama_pos startPos,
      llama_pos endPos) const = 0;
  virtual void seqAdd(
      ContextSliderMemoryHandle mem, llama_seq_id seqId, llama_pos startPos,
      llama_pos endPos, llama_pos delta) const = 0;
};

/// Returns the default llama-backed ops implementation.
const IContextSliderOps& defaultContextSliderOps();

struct ContextUsage {
  llama_pos pos = 0;
  llama_pos cacheTokens = 0;
};

/// Outcome of a sliding-window operation on the KV cache.
struct ContextSlideOutcome {
  enum class Kind {
    NotNeeded, // Context had enough room; no slide performed
    Slid,      // Successfully discarded tokens via partial slide
    FullWipe,  // Fallback: wiped everything after firstMsgTokens (prefill only)
    Overflow,  // Could not free enough space; caller should throw
    MemoryOperationFailed, // llama memory rejected the requested slide
  };

  Kind kind = Kind::NotNeeded;
  llama_pos newNPast = 0;  // Updated nPast after the slide
  llama_pos discarded = 0; // Number of tokens actually discarded
};

/// Attempts to slide the context window during prefill (eval) phase.
///
/// This handles the case where adding nTokensToAppend would overflow the
/// context. It tries to discard tokens from the middle (after firstMsgTokens)
/// while respecting the tools_compact anchor via ToolsCompactController.
///
/// On success (Slid or FullWipe), the KV cache has been modified and newNPast
/// reflects the new position. On NotNeeded, no action was taken. On Overflow,
/// the caller should throw a context overflow error.
///
/// @param lctx           The llama context for KV cache operations
/// @param seqId          The llama sequence to slide
/// @param nPast          Current token position in the context
/// @param firstMsgTokens Number of tokens in the first message (protected)
/// @param nTokensToAppend Number of tokens about to be appended
/// @param nDiscarded     Maximum tokens the caller allows to discard
/// @param tools          Controller for tools_compact anchor management
/// @param ops            Indirection over llama context/memory operations
/// @param effectiveCtx   Per-sequence token ceiling to slide against. When
///                       <= 0, falls back to the whole-context size reported
///                       by ops.nCtx(). In batch mode this is the partitioned
///                       per-slot cap (ctx / n_parallel), which is smaller
///                       than the full context.
/// @return ContextSlideOutcome describing what happened and the new state
ContextSlideOutcome trySlidePrefill(
    llama_context* lctx, llama_seq_id seqId, llama_pos nPast,
    llama_pos firstMsgTokens, llama_pos nTokensToAppend, llama_pos nDiscarded,
    ToolsCompactController& tools,
    const IContextSliderOps& ops = defaultContextSliderOps(),
    llama_pos effectiveCtx = -1);

ContextSlideOutcome trySlidePrefill(
    llama_context* lctx, llama_seq_id seqId, ContextUsage current,
    ContextUsage protectedPrefix, ContextUsage append, llama_pos nDiscarded,
    ToolsCompactController& tools,
    const IContextSliderOps& ops = defaultContextSliderOps());

/// Attempts to slide the context window during generation phase.
///
/// This handles the case where generating one more token would overflow the
/// context. Unlike prefill, there is no FullWipe fallback during generation.
/// If sliding cannot free space, returns NotNeeded with no action.
///
/// @param lctx           The llama context for KV cache operations
/// @param seqId          The llama sequence to slide
/// @param nPast          Current token position in the context
/// @param firstMsgTokens Number of tokens in the first message (protected)
/// @param nDiscarded     Maximum tokens the caller allows to discard
/// @param tools          Controller for tools_compact anchor management
/// @param ops            Indirection over llama context/memory operations
/// @param effectiveCtx   Per-sequence token ceiling to slide against. When
///                       <= 0, falls back to the whole-context size reported
///                       by ops.nCtx() (single-sequence behaviour).
/// @param nCacheTokens   Actual KV-cache occupancy when it differs from nPast
///                       (e.g. multimodal). <= -1 means it equals nPast.
/// @return ContextSlideOutcome describing what happened and the new state
ContextSlideOutcome trySlideGeneration(
    llama_context* lctx, llama_seq_id seqId, llama_pos nPast,
    llama_pos firstMsgTokens, llama_pos nDiscarded,
    ToolsCompactController& tools,
    const IContextSliderOps& ops = defaultContextSliderOps(),
    llama_pos effectiveCtx = -1, llama_pos nCacheTokens = -1);

/// Outcome of an in-place KV-cache range compaction.
struct CompactRangeOutcome {
  enum class Kind {
    NoOp,                  // Empty / inverted range; cache untouched
    Compacted,             // Range removed and tail shifted
    MemoryOperationFailed, // seqRm rejected the request
  };

  Kind kind = Kind::NoOp;
  llama_pos newNPast = 0;
  llama_pos discarded = 0;
};

/// Drops `[startPos, endPos)` from `seqId`'s KV cache and shifts the tail
/// `[endPos, nPast)` down. Pure primitive; the caller owns policy
/// (overflow checks, `firstMsgTokens` / tools_compact accounting).
/// Returns NoOp for empty / out-of-range inputs without touching the cache.
CompactRangeOutcome compactKvRange(
    llama_context* lctx, llama_seq_id seqId, llama_pos startPos,
    llama_pos endPos, llama_pos nPast,
    const IContextSliderOps& ops = defaultContextSliderOps());

/// Per-span outcome record produced by `runReasoningCompaction`. Kept so
/// callers (production: `TextLlmContext`) can emit per-span logs without
/// pulling logging into the orchestration core. `nPastAfter` /
/// `firstMsgTokensAfter` capture the running state right after this
/// span's compaction so per-span log output matches the pre-refactor
/// hand-rolled loop.
struct ReasoningCompactionEntry {
  llama_pos start = 0;
  llama_pos end = 0;
  enum class Status {
    Skipped,               // end < 0 (still open) or end <= start (degenerate)
    Compacted,             // KV range removed, tail shifted
    MemoryOperationFailed, // seqRm rejected the request
    NoOp,                  // compactOp returned NoOp (defensive guard)
  };
  Status status = Status::Skipped;
  llama_pos discarded = 0;
  // Running state right after this entry was processed. Only meaningful
  // for Compacted entries; for Skipped / Failed / NoOp these mirror the
  // pre-step state (nothing changed).
  llama_pos nPastAfter = 0;
  llama_pos firstMsgTokensAfter = 0;
};

/// Aggregated result of `runReasoningCompaction`: updated state values plus
/// the per-span entries.
struct ReasoningCompactionResult {
  llama_pos newNPast = 0;
  llama_pos newFirstMsgTokens = 0;
  int32_t discardsApplied = 0;
  std::vector<ReasoningCompactionEntry> entries;
};

/// Callable for per-span KV compaction. Production wires this to
/// `compactKvRange`; tests inject deterministic fakes.
using ReasoningCompactionOp = std::function<
    CompactRangeOutcome(llama_pos start, llama_pos end, llama_pos nPast)>;

/// Callable for tools_compact-style controller notification. Production
/// wires this to `ToolsCompactController::onSlide`; tests record args.
using ReasoningSlideNotifier =
    std::function<void(int discarded, llama_pos start)>;

/// Walk `spans` in reverse and compact each via `compactOp`. On each
/// successful compaction:
///   - `nPast` shrinks by `discarded`
///   - `firstMsgTokens` clamps down to `start` if `start < firstMsgTokens`
///   - `onSlide(discarded, start)` is invoked (callers gate on whether the
///     downstream controller is enabled)
///   - `discardsApplied` increments
/// Spans with `end < 0` (still open) or `end <= start` (degenerate) are
/// skipped without invoking `compactOp` — the single backstop for all the
/// close-capture sites in TextLlmContext.
///
/// Reverse iteration is required: compacting a later span shifts no
/// earlier-span positions because the operation only renumbers tokens
/// after `end`. Walking forwards would invalidate the recorded `start`
/// of every span after the first.
ReasoningCompactionResult runReasoningCompaction(
    const std::vector<std::pair<llama_pos, llama_pos>>& spans, llama_pos nPast,
    llama_pos firstMsgTokens, const ReasoningCompactionOp& compactOp,
    const ReasoningSlideNotifier& onSlide);

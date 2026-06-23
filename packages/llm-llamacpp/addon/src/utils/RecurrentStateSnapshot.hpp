#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <llama.h>

namespace qvac_lib_inference_addon_llama {
namespace utils {

// Owning byte buffer holding a full-state snapshot of a single
// sequence. Captured via `llama_state_seq_get_data_ext` with
// `flags == 0`, which on hybrid memories (Qwen3.5 / Qwen3-Next /
// Jamba / ...) covers BOTH the attention KV and the recurrent (SSM /
// RWKV) hidden state.
//
// Why full state: a partial-only snapshot rolls back only the
// recurrent half, and restoring it leaves the attention KV ahead with
// the post-snapshot tail still present. Trimming that tail with
// `seq_rm` is rejected by the recurrent memory module
// (`llama_memory_recurrent::seq_rm`) for any partial-tail range that
// includes the final committed position — Mamba/RWKV-style state is
// not reversible, so the API only accepts full clears or
// non-overlapping ranges. A full-state snapshot sidesteps the issue:
// `state_seq_set_data_ext(flags=0)` rebuilds the entire sequence in
// one shot, so no `seq_rm` is needed.
//
// `nPast` records the next-position-to-write at snapshot time. The
// caller uses it as the replay anchor and the post-restore `nPast_`.
struct RecurrentStateSnapshot {
  std::vector<uint8_t> data;
  llama_pos nPast = 0;

  [[nodiscard]] bool empty() const noexcept { return data.empty(); }
  [[nodiscard]] size_t size() const noexcept { return data.size(); }
  void clear() noexcept {
    data.clear();
    nPast = 0;
  }
};

// Captures the full state of `seqId` into `out`, recording `nPastAt`
// alongside the data. Calls `llama_state_seq_get_size_ext(..., 0)`
// and, if size > 0, `llama_state_seq_get_data_ext(..., 0)`.
//
// Returns true on success. Returns false when the data fetch
// underflows the reported size — `out` is cleared so it cannot be
// partially restored later.
bool snapshotRecurrentState(
    ::llama_context* lctx, llama_seq_id seqId, llama_pos nPastAt,
    RecurrentStateSnapshot& out);

// Restores `snapshot` into `seqId`, fully replacing the sequence's
// attention KV and recurrent state. No-op when `snapshot` is empty.
// Returns true on success, false when the underlying
// `llama_state_seq_set_data_ext` reports a short read.
bool restoreRecurrentState(
    ::llama_context* lctx, llama_seq_id seqId,
    const RecurrentStateSnapshot& snapshot);

// Replays `tokens` through `lctx` against `seqId`, attaching them to
// positions starting at `startPos` (so position[i] == startPos + i).
// Used after a partial-state restore to advance the recurrent state
// across the post-reasoning span without re-running the sampler. The
// batch is chunked to fit within `llama_n_batch(lctx)` so callers can
// pass arbitrarily long token vectors.
//
// `outputLogitsForLast` controls whether the final token in `tokens`
// requests output logits from `llama_decode` — set true when the
// caller intends to immediately sample the next token from the
// post-replay state, false when the replay is purely for SSM advance.
//
// Returns true on success. Returns false if any sub-batch decode call
// reports a non-zero error code; the caller should treat the recurrent
// state as undefined in that case (the attention KV the caller
// previously compacted is unaffected).
bool replayTokensThroughDecoder(
    ::llama_context* lctx, llama_seq_id seqId,
    const std::vector<llama_token>& tokens, llama_pos startPos,
    bool outputLogitsForLast = false);

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

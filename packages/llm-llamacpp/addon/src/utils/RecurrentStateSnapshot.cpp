#include "RecurrentStateSnapshot.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <common/common.h>
#include <llama.h>

namespace qvac_lib_inference_addon_llama {
namespace utils {

bool snapshotRecurrentState(
    ::llama_context* lctx, llama_seq_id seqId, llama_pos nPastAt,
    RecurrentStateSnapshot& out) {
  out.clear();
  if (lctx == nullptr) {
    return false;
  }

  // Full-state snapshot (flags = 0). On hybrid memories this captures
  // both the attention KV and the recurrent state for `seqId`, so a
  // later `state_seq_set_data_ext` can rebuild the sequence in one
  // call without needing `seq_rm` (which the recurrent module rejects
  // for partial-tail ranges that include the final committed pos).
  const size_t fullSize = llama_state_seq_get_size_ext(lctx, seqId, /*flags=*/0);
  if (fullSize == 0) {
    // Empty sequence (no committed tokens) — successful no-op
    // snapshot. Restoring it later is also a no-op.
    out.nPast = nPastAt;
    return true;
  }

  out.data.resize(fullSize);
  const size_t written = llama_state_seq_get_data_ext(
      lctx, out.data.data(), fullSize, seqId, /*flags=*/0);
  if (written != fullSize) {
    out.clear();
    return false;
  }
  out.nPast = nPastAt;
  return true;
}

bool restoreRecurrentState(
    ::llama_context* lctx, llama_seq_id seqId,
    const RecurrentStateSnapshot& snapshot) {
  if (lctx == nullptr) {
    return false;
  }
  if (snapshot.empty()) {
    return true;
  }
  const size_t read = llama_state_seq_set_data_ext(
      lctx, snapshot.data.data(), snapshot.data.size(), seqId, /*flags=*/0);
  return read == snapshot.data.size();
}

bool replayTokensThroughDecoder(
    ::llama_context* lctx, llama_seq_id seqId,
    const std::vector<llama_token>& tokens, llama_pos startPos,
    bool outputLogitsForLast) {
  if (tokens.empty()) {
    return true;
  }
  if (lctx == nullptr) {
    return false;
  }

  // Chunk the replay so it fits within the context's micro-batch
  // capacity. `llama_n_batch` returns the logical batch size; we use
  // it as an upper bound on `common_batch_add` calls per `llama_decode`.
  const auto nBatchU = llama_n_batch(lctx);
  if (nBatchU == 0) {
    return false;
  }
  const int32_t chunkSize = static_cast<int32_t>(nBatchU);
  const int32_t total = static_cast<int32_t>(tokens.size());

  llama_batch batch = llama_batch_init(chunkSize, 0, 1);
  bool ok = true;
  for (int32_t offset = 0; offset < total && ok; offset += chunkSize) {
    const int32_t end = std::min(offset + chunkSize, total);
    common_batch_clear(batch);
    for (int32_t i = offset; i < end; ++i) {
      const bool isFinal = (i == total - 1);
      const bool requestLogits = outputLogitsForLast && isFinal;
      common_batch_add(
          batch,
          tokens[i],
          startPos + static_cast<llama_pos>(i),
          {seqId},
          requestLogits);
    }
    if (llama_decode(lctx, batch) != 0) {
      ok = false;
    }
  }
  llama_batch_free(batch);
  return ok;
}

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

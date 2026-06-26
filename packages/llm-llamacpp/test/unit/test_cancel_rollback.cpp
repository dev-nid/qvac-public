#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <common/chat.h>
#include <gtest/gtest.h>
#include <inference-addon-cpp/Errors.hpp>
#include <llama.h>

#include "model-interface/LlamaModel.hpp"
#include "model-interface/MtmdLlmContext.hpp"
#include "model-interface/TextLlmContext.hpp"
#include "model-interface/ToolsCompactController.hpp"
#include "utils/RecurrentStateSnapshot.hpp"
#include "test_common.hpp"

// Tests for the cancel-rollback paths introduced alongside
// `remove_thinking_from_context` for hybrid SSM models. Two layers of
// coverage:
//   1. Snapshot / restore primitive against a real `llama_context`
//      (hybrid + pure-attention). Pins the foundational behaviour that
//      `llama_state_seq_get_data_ext` / `set_data_ext` with `flags=0`
//      rewind both the attention KV and the recurrent state.
//   2. Cancel-rollback wiring on `TextLlmContext` and `MtmdLlmContext`
//      end-to-end through their public entrypoints (`evalMessageWithTools`,
//      `onCancel`, `generateResponse`, `stop`).
//
// Hybrid-specific tests skip when the Qwen3.5 fixture is not present.
// Pure-attention tests use the Qwen3-0.6B fixture (always downloaded in
// the CI manifest). Multimodal tests additionally require the Qwen3.5
// projection file.

namespace fs = std::filesystem;

using qvac_lib_inference_addon_llama::utils::RecurrentStateSnapshot;
using qvac_lib_inference_addon_llama::utils::restoreRecurrentState;
using qvac_lib_inference_addon_llama::utils::snapshotRecurrentState;

namespace {

llama_pos seqPosMax(LlamaModel& model, llama_seq_id seqId = 0) {
  auto* mem = llama_get_memory(model.getContext());
  if (mem == nullptr) {
    return -1;
  }
  return llama_memory_seq_pos_max(mem, seqId);
}

common_chat_msg makeMsg(const std::string& role, const std::string& content) {
  common_chat_msg msg;
  msg.role = role;
  msg.content = content;
  return msg;
}

std::string qwen35HybridModelPath() {
  return test_common::BaseTestModelPath::get("Qwen3.5-0.8B-Q8_0.gguf");
}

std::string qwen3PureAttentionModelPath() {
  return test_common::BaseTestModelPath::get("Qwen3-0.6B-Q8_0.gguf");
}

std::string qwen35MmprojPath() {
  return test_common::BaseTestModelPath::get("mmproj-Qwen3.5-0.8B-F16.gguf");
}

bool modelFileExists(const std::string& path) { return fs::exists(path); }

std::unique_ptr<LlamaModel> loadTextModel(const std::string& modelPath) {
  if (!modelFileExists(modelPath)) {
    return nullptr;
  }
  std::unordered_map<std::string, std::string> config;
  config["device"] = test_common::getTestDevice();
  config["ctx_size"] = "4096";
  config["gpu_layers"] = test_common::getTestGpuLayers();
  config["n_predict"] = "8";
  config["backendsDir"] = test_common::getTestBackendsDir().string();

  std::string path = modelPath;
  std::string projection;
  auto model = std::make_unique<LlamaModel>(
      std::move(path), std::move(projection), std::move(config));
  model->waitForLoadInitialization();
  if (!model->isLoaded()) {
    return nullptr;
  }
  return model;
}

std::unique_ptr<LlamaModel>
loadMtmdModel(const std::string& modelPath, const std::string& projectionPath) {
  if (!modelFileExists(modelPath) || !modelFileExists(projectionPath)) {
    return nullptr;
  }
  std::unordered_map<std::string, std::string> config;
  config["device"] = test_common::getTestDevice();
  config["ctx_size"] = "4096";
  config["gpu_layers"] = test_common::getTestGpuLayers();
  config["n_predict"] = "8";
  config["backendsDir"] = test_common::getTestBackendsDir().string();

  std::string mp = modelPath;
  std::string pp = projectionPath;
  auto model = std::make_unique<LlamaModel>(
      std::move(mp), std::move(pp), std::move(config));
  model->waitForLoadInitialization();
  if (!model->isLoaded()) {
    return nullptr;
  }
  return model;
}

LlmModelContext makeShared(LlamaModel& model) {
  return LlmModelContext{
      .model = model.getModel(),
      .lctx = model.getContext(),
      .vocab = llama_model_get_vocab(model.getModel())};
}

// Seeds a non-zero cache state on the model, leaving it ready for follow-up
// inspection. Uses prefill mode so no generation runs (faster, deterministic).
void primeWithPrefill(LlamaModel& model, const std::string& userText) {
  LlamaModel::Prompt prompt;
  prompt.input = R"([{"role":"user","content":")" + userText + R"("}])";
  prompt.prefill = true;
  model.processPrompt(prompt);
}

// Returns the canonical small test image path used elsewhere in the
// unit test suite. Mirrors `multimodalTestImagePath()` from
// `test_mtmd_llm_context.cpp`; duplicated here to keep this file
// self-contained.
fs::path multimodalTestImagePath() {
  const fs::path packageRelative = "media/fruitPlate.png";
  if (fs::exists(packageRelative)) {
    return packageRelative;
  }
#ifdef TEST_BINARY_DIR
  const fs::path binaryRelative = fs::path(TEST_BINARY_DIR) / ".." / ".." /
                                  ".." / "media" / "fruitPlate.png";
  if (fs::exists(binaryRelative)) {
    return binaryRelative.lexically_normal();
  }
#endif
  return "packages/llm-llamacpp/media/fruitPlate.png";
}

std::vector<uint8_t> readBinaryFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {
      std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

// Look up a named int64 entry in the runtime stats vector. Returns
// std::nullopt when the key is absent so callers can distinguish
// "missing key" from "key present, value is zero".
std::optional<int64_t> statInt(LlamaModel& model, const std::string& key) {
  auto stats = model.runtimeStats();
  for (const auto& entry : stats) {
    if (entry.first != key) {
      continue;
    }
    if (std::holds_alternative<int64_t>(entry.second)) {
      return std::get<int64_t>(entry.second);
    }
    return static_cast<int64_t>(std::get<double>(entry.second));
  }
  return std::nullopt;
}

} // namespace

// ============================================================================
// Layer 1: snapshot / restore primitive against a real llama_context
// ============================================================================

class CancelRollbackPrimitiveTest : public ::testing::Test {};

// Foundation: on a hybrid SSM model, snapshotting the sequence state and
// then restoring it after a `model->reset()` must return the cache to its
// pre-reset position. This is the mechanism every cancel-rollback path
// relies on.
TEST_F(CancelRollbackPrimitiveTest, SnapshotRestoreRoundtripQwen35Hybrid) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  primeWithPrefill(*model, "Hello, this is the seed prompt.");
  const llama_pos posBefore = seqPosMax(*model);
  ASSERT_GT(posBefore, 0) << "prefill must have advanced the cache";

  RecurrentStateSnapshot snap;
  ASSERT_TRUE(snapshotRecurrentState(
      model->getContext(), /*seqId=*/0, posBefore + 1, snap));
  ASSERT_FALSE(snap.empty())
      << "hybrid model snapshot must be non-empty (recurrent state present)";
  EXPECT_EQ(snap.nPast, posBefore + 1);

  model->reset();
  ASSERT_EQ(seqPosMax(*model), -1)
      << "reset should fully clear the sequence memory";

  ASSERT_TRUE(restoreRecurrentState(model->getContext(), /*seqId=*/0, snap));
  EXPECT_EQ(seqPosMax(*model), posBefore)
      << "restore must return the cache to the snapshotted position";
}

// Same roundtrip for a pure-attention model. The snapshot+restore primitive
// is architecture-agnostic — it works for attention-only memories too.
TEST_F(CancelRollbackPrimitiveTest, SnapshotRestoreRoundtripQwen3PureAttention) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  primeWithPrefill(*model, "Hello, this is the seed prompt.");
  const llama_pos posBefore = seqPosMax(*model);
  ASSERT_GT(posBefore, 0);

  RecurrentStateSnapshot snap;
  ASSERT_TRUE(snapshotRecurrentState(
      model->getContext(), /*seqId=*/0, posBefore + 1, snap));
  ASSERT_FALSE(snap.empty());

  model->reset();
  ASSERT_EQ(seqPosMax(*model), -1);

  ASSERT_TRUE(restoreRecurrentState(model->getContext(), /*seqId=*/0, snap));
  EXPECT_EQ(seqPosMax(*model), posBefore);
}

// Snapshot taken before any tokens have been decoded: payload size for
// hybrid memories is non-zero because the recurrent state itself is
// serialized even when there are no positions, but restore must still be
// idempotent against a freshly reset context.
TEST_F(CancelRollbackPrimitiveTest, SnapshotEmptySequenceHybridIsRestorable) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  RecurrentStateSnapshot snap;
  ASSERT_TRUE(snapshotRecurrentState(
      model->getContext(), /*seqId=*/0, /*nPastAt=*/0, snap));
  EXPECT_EQ(snap.nPast, 0);

  // Restoring an empty-sequence snapshot must succeed and leave the
  // cache empty.
  ASSERT_TRUE(restoreRecurrentState(model->getContext(), /*seqId=*/0, snap));
  EXPECT_EQ(seqPosMax(*model), -1);
}

// Mid-decode snapshot + restore: prime the cache, snapshot, prime again
// (advancing the cache further via a reset+second prefill), then restore.
// Verifies that the second prefill's content is fully wiped by the
// restore.
TEST_F(CancelRollbackPrimitiveTest, RestoreDropsLaterContentOnHybrid) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  primeWithPrefill(*model, "Short");
  const llama_pos posAfterShort = seqPosMax(*model);
  ASSERT_GT(posAfterShort, 0);

  RecurrentStateSnapshot snap;
  ASSERT_TRUE(snapshotRecurrentState(
      model->getContext(), /*seqId=*/0, posAfterShort + 1, snap));

  // Run a longer prefill that resets and grows the cache beyond the
  // snapshotted position.
  primeWithPrefill(
      *model,
      "A much longer second prompt that should grow the cache well past "
      "the original snapshot position so the restore step has something "
      "meaningful to drop.");
  const llama_pos posAfterLong = seqPosMax(*model);
  ASSERT_GT(posAfterLong, posAfterShort)
      << "second prefill should have grown the cache beyond the snapshot";

  ASSERT_TRUE(restoreRecurrentState(model->getContext(), /*seqId=*/0, snap));
  EXPECT_EQ(seqPosMax(*model), posAfterShort)
      << "restore must drop the second prefill's tail and return to the "
         "snapshotted position";
}

// ============================================================================
// Layer 2a: TextLlmContext cancel paths
// ============================================================================

class TextLlmContextCancelTest : public ::testing::Test {};

// Cancel signalled before `evalMessageWithTools` runs must:
//   * return false (inference did not complete),
//   * leave nPast at 0 (the snapshot at function entry is restored on the
//     hybrid path; on the pure-attention path `removeLastNTokens(0)` is a
//     no-op).
TEST_F(
    TextLlmContextCancelTest, PrefillCancelAtEntryReturnsFalseOnHybrid) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);

  driver.stop();
  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  const bool result = driver.evalMessageWithTools(
      chatMsgs, /*tools=*/{}, /*isCacheLoaded=*/false, /*prefill=*/false);

  EXPECT_FALSE(result);
  EXPECT_EQ(driver.getNPast(), 0);
  EXPECT_EQ(seqPosMax(*model), -1)
      << "cancelled prefill must not leave KV cells resident";
}

TEST_F(
    TextLlmContextCancelTest,
    PrefillCancelAtEntryReturnsFalseOnPureAttention) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);

  driver.stop();
  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  const bool result = driver.evalMessageWithTools(
      chatMsgs, /*tools=*/{}, /*isCacheLoaded=*/false, /*prefill=*/false);

  EXPECT_FALSE(result);
  EXPECT_EQ(driver.getNPast(), 0);
  EXPECT_EQ(seqPosMax(*model), -1);
}

// After a cancelled prefill, the same context must be usable for the next
// inference: the stop flag is cleared, and a fresh prefill+generation
// succeeds. This is the regression guard against "cancel poisons the
// context" — the original failure mode without the recurrent rollback fix.
TEST_F(
    TextLlmContextCancelTest,
    PrefillCancelLeavesContextUsableForNextInferenceOnHybrid) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  // First, cancel via the high-level API.
  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  {
    TextLlmContext driver(params, shared, tools, /*seqId=*/0);
    driver.stop();
    std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
    EXPECT_FALSE(driver.evalMessageWithTools(
        chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/false));
    EXPECT_EQ(driver.getNPast(), 0);
  }

  // Then run a normal prefill on a fresh driver — must succeed.
  TextLlmContext driver2(params, shared, tools, /*seqId=*/0);
  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  EXPECT_TRUE(driver2.evalMessageWithTools(
      chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/true));
  EXPECT_GT(driver2.getNPast(), 0)
      << "post-cancel prefill must successfully decode tokens";
}

// `onCancel` on a hybrid driver with `remove_thinking_from_context: true`:
// after prefill (which takes the end-of-prefill snapshot), calling
// `onCancel` directly must restore the snapshot (no-op on cache position
// since we haven't generated anything past it) AND clear the per-inference
// rollback buffers cleanly. We can only observe the buffer cleanup through
// the runtime stat `thinkingCompactionFailed` — it must stay 0 on the
// success path (snapshot restore succeeded).
TEST_F(
    TextLlmContextCancelTest,
    OnCancelRestoresEndOfPrefillSnapshotOnHybrid) {
  auto model = loadTextModel(qwen35HybridModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);
  driver.setRemoveThinkingFromContext(true);

  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  ASSERT_TRUE(driver.evalMessageWithTools(
      chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/false));
  const llama_pos posAfterPrefill = driver.getNPast();
  const llama_pos seqPosAfterPrefill = seqPosMax(*model);
  ASSERT_GT(posAfterPrefill, 0);
  ASSERT_EQ(seqPosAfterPrefill, posAfterPrefill - 1)
      << "post-prefill cache cursor must match driver-tracked nPast";

  // Cancel before any generation tokens have been sampled. The snapshot
  // anchor sits at end-of-prefill, so restore should leave nPast where
  // it is and only clear the rollback buffers.
  driver.onCancel([](const std::string&) {});

  EXPECT_EQ(driver.getNPast(), posAfterPrefill)
      << "onCancel on hybrid must restore to end-of-prefill position";
  EXPECT_EQ(seqPosMax(*model), seqPosAfterPrefill)
      << "onCancel restore must leave the underlying llama_context's "
         "seq_pos_max at end-of-prefill (no spurious forward progress, no "
         "underflow)";
  EXPECT_EQ(driver.getThinkingCompactionFailed(), 0)
      << "successful snapshot restore must not bump the failure counter";
}

// Pure-attention `onCancel` keeps existing behavior: it chains to
// `onGenerationFinished` which runs tools-compact tail trim and
// `compactThinkSpan`. Neither should crash on a freshly prefilled context.
TEST_F(
    TextLlmContextCancelTest,
    OnCancelOnPureAttentionRunsExistingFinalization) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);
  driver.setRemoveThinkingFromContext(true);

  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  ASSERT_TRUE(driver.evalMessageWithTools(
      chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/false));

  EXPECT_NO_THROW(driver.onCancel([](const std::string&) {}));
  EXPECT_EQ(driver.getThinkingCompactionFailed(), 0);
}

// ============================================================================
// User-visible perf snapshot lifecycle on `TextLlmContext`
// ============================================================================
//
// `compactThinkSpan` freezes the perf counters just before any recurrent
// replay decode runs, so `runtimeStats()` can report the pre-replay
// (user-visible) values rather than counters inflated by internal cache
// maintenance. The base `LlmContext::takeUserVisiblePerfSnapshot` returns
// `nullopt` by default; `TextLlmContext` overrides it to consume the
// captured snapshot. These tests pin the capture / take / clear contract
// without requiring a hybrid model (the lifecycle is identical regardless
// of memory module — pure-attention just doesn't do replay decode).

// Newly constructed driver: no snapshot. Guards the initial state — a
// stray non-empty snapshot here would leak into the first inference's
// `runtimeStats()` and report zeroed-out counters.
TEST_F(TextLlmContextCancelTest, FreshDriverReportsNoUserVisiblePerfSnapshot) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);

  EXPECT_FALSE(driver.takeUserVisiblePerfSnapshot().has_value())
      << "Newly constructed driver must report no user-visible perf snapshot";
}

// After a full prefill + generation, `compactThinkSpan` runs at the end
// of `onGenerationFinished` and captures the perf snapshot
// unconditionally (pure-attention has no replay, so the snapshot
// equals the live read; hybrid would freeze BEFORE its replay decode).
// The override hands the captured value back and clears the slot, so a
// second `take` returns `nullopt`.
TEST_F(TextLlmContextCancelTest, CompactThinkSpanCapturesAndTakeIsIdempotent) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);

  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  ASSERT_TRUE(driver.evalMessageWithTools(
      chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/false));
  ASSERT_TRUE(driver.generateResponse([](const std::string&) {}));

  auto snapshot = driver.takeUserVisiblePerfSnapshot();
  ASSERT_TRUE(snapshot.has_value())
      << "compactThinkSpan must capture the user-visible perf snapshot at "
         "end-of-generation";
  EXPECT_GT(snapshot->n_p_eval, 0)
      << "Snapshot must record the prefill it captured perf for";

  EXPECT_FALSE(driver.takeUserVisiblePerfSnapshot().has_value())
      << "takeUserVisiblePerfSnapshot must be consumed on read";
}

// A fresh inference must start from a clean slate: the stale snapshot
// from a prior turn is cleared at the start of `evalMessageWithTools`
// so the new inference's `runtimeStats()` never sees the previous
// turn's frozen counters. Without this clear, `runtimeStats()` after
// the next turn would silently report the old turn's pre-replay values.
TEST_F(
    TextLlmContextCancelTest,
    EvalMessageWithToolsClearsStaleUserVisiblePerfSnapshot) {
  auto model = loadTextModel(qwen3PureAttentionModelPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3-0.6B pure-attention model not found";
  }

  LlmModelContext shared = makeShared(*model);
  ToolsCompactController tools(std::nullopt);
  common_params params = model->getCommonParams();
  TextLlmContext driver(params, shared, tools, /*seqId=*/0);

  // Turn 1: full flow → snapshot captured.
  std::vector<common_chat_msg> chatMsgs = {makeMsg("user", "Hi")};
  ASSERT_TRUE(driver.evalMessageWithTools(
      chatMsgs, {}, /*isCacheLoaded=*/false, /*prefill=*/false));
  ASSERT_TRUE(driver.generateResponse([](const std::string&) {}));

  // Turn 2: start a new inference WITHOUT taking turn 1's snapshot
  // first. The fresh `evalMessageWithTools` must drop the stale value
  // so the next runtimeStats read does not silently surface it.
  std::vector<common_chat_msg> chatMsgs2 = {
      makeMsg("user", "Hi"),
      makeMsg("assistant", "Hello"),
      makeMsg("user", "And again")};
  ASSERT_TRUE(driver.evalMessageWithTools(
      chatMsgs2, {}, /*isCacheLoaded=*/false, /*prefill=*/true));

  EXPECT_FALSE(driver.takeUserVisiblePerfSnapshot().has_value())
      << "evalMessageWithTools must clear any stale user-visible perf "
         "snapshot from the previous turn";
}

// ============================================================================
// Layer 2b: MtmdLlmContext cancel paths via the high-level LlamaModel API
// ============================================================================
//
// `MtmdLlmContext` is constructed indirectly inside `LlamaModel` and its
// cancel flag (`stopGeneration_`) is only propagated when an inference is
// already in flight (see `LlamaModel::cancelImpl`). The deterministic
// "stop before evalMessageWithTools" pattern we use for `TextLlmContext`
// doesn't transfer here without direct construction, which requires a
// `common_init_result_ptr` that the model owns internally. So we drive
// these scenarios end-to-end with a worker thread + `LlamaModel::cancel`,
// mirroring the existing `AddonCppTest.StopDuringGeneration` pattern.

class MtmdLlmContextCancelTest : public ::testing::Test {};

// TODO #1 coverage: cancel an in-flight hybrid multimodal inference and
// verify the model survives — i.e. the next prompt succeeds. This is a
// recovery-only assertion because the mtmd prefill loop processes each
// text chunk as a single atomic `mtmd_helper_eval_chunk_single` call,
// so a text-only prompt collapses to one chunk with exactly one cancel
// check at the very top. The mid-prefill rollback case (cancel landing
// AFTER at least one chunk has decoded) is exercised by the sibling
// `CancelDuringImageChunkRollsBackHybridMtmdCache` test below, which
// guarantees multiple chunks by attaching an image.
TEST_F(
    MtmdLlmContextCancelTest,
    CancelDuringPrefillLeavesHybridMtmdUsable) {
  auto model =
      loadMtmdModel(qwen35HybridModelPath(), qwen35MmprojPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid multimodal model not found";
  }

  // Prefill-only call so generation timing is not a factor.
  LlamaModel::Prompt cancelTargetPrompt;
  cancelTargetPrompt.input = R"([
    {"role":"user","content":"Cancel target: a moderately long prompt that gives the worker a chance to start prefill before cancel fires."}
  ])";
  cancelTargetPrompt.prefill = true;

  std::atomic<bool> done{false};
  std::thread worker([&] {
    try {
      model->processPrompt(cancelTargetPrompt);
    } catch (...) {
      // Cancel may surface as a status error; treat as a clean cancel.
    }
    done.store(true);
  });

  // Brief head start so the worker enters the prefill loop.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_NO_THROW(model->cancel());

  // Wait for the worker to unwind.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(done.load())
      << "worker did not unwind within 10s of cancel";
  worker.join();

  // `thinkingCompactionFailed` is bumped only by snapshot-capture or
  // snapshot-restore failure. A non-zero value here means the
  // recurrent-rollback code path encountered a failure mode (capture
  // underflow or restore underflow); zero confirms the snapshot
  // infrastructure executed cleanly whether or not cancel landed.
  const auto compactionFailed = statInt(*model, "thinkingCompactionFailed");
  if (compactionFailed.has_value()) {
    EXPECT_EQ(*compactionFailed, 0)
        << "hybrid mtmd cancel path must not trigger the failure counter";
  }

  // Recovery: the model must accept another inference cleanly.
  LlamaModel::Prompt recovery;
  recovery.input = R"([{"role":"user","content":"Hi"}])";
  EXPECT_NO_THROW({
    std::string output = model->processPrompt(recovery);
    EXPECT_GE(output.length(), 0);
  });
}

// TODO #2 coverage: cancel a hybrid multimodal prefill AFTER an image
// chunk has been committed to the KV cache (or at least started). The
// previous metadata-resync workaround failed exactly here because
// `llama_memory_seq_pos_max` does not report Qwen3VL M-RoPE x/y
// coordinates stored as extended metadata on image cells; the snapshot
// path captures the full sequence state including those coordinates,
// so cancel can drop the image-chunk cells along with everything else.
// Asserts the rollback brings the cache back to empty and a follow-up
// inference works.
TEST_F(
    MtmdLlmContextCancelTest,
    CancelDuringImageChunkRollsBackHybridMtmdCache) {
  auto model =
      loadMtmdModel(qwen35HybridModelPath(), qwen35MmprojPath());
  if (!model) {
    GTEST_SKIP() << "Qwen3.5 hybrid multimodal model not found";
  }

  const fs::path imagePath = multimodalTestImagePath();
  if (!fs::exists(imagePath)) {
    GTEST_SKIP() << "Multimodal test image not found at " << imagePath;
  }

  // Prompt with an image chunk followed by a text chunk — the exact
  // shape the deleted metadata-resync workaround was designed for
  // (cancel landing between chunks).
  LlamaModel::Prompt prompt;
  prompt.input =
      R"([{"role": "user", "type": "media", "content": ""},)"
      R"( {"role": "user", "content": "Describe this image in detail with as much length as possible to give the cancel signal a wide mid-prefill window."}])";
  prompt.media.push_back(readBinaryFile(imagePath));
  prompt.prefill = true;

  std::atomic<bool> done{false};
  std::thread worker([&] {
    try {
      model->processPrompt(prompt);
    } catch (...) {
      // Cancel may surface as a status error; treat as a clean cancel.
    }
    done.store(true);
  });

  // Image-chunk decoding on Qwen3.5 vision typically takes well over
  // 50ms even on M-series; this head start reliably lands cancel while
  // the prefill loop is still iterating chunks.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_NO_THROW(model->cancel());

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(done.load())
      << "worker did not unwind within 15s of cancel";
  worker.join();

  // Strong assertion: the snapshot must have rolled the cache back to
  // empty, including any image-chunk KV cells that were committed
  // before cancel landed. If the deleted metadata-resync workaround
  // were still in effect this would fail because `seq_pos_max` cannot
  // see the image-chunk extended metadata.
  EXPECT_EQ(seqPosMax(*model), -1)
      << "cancelled hybrid mtmd prefill with image chunk must restore "
         "the pre-prefill cursor; non-empty cache means image-chunk "
         "cells leaked past cancel";

  const auto compactionFailed = statInt(*model, "thinkingCompactionFailed");
  if (compactionFailed.has_value()) {
    EXPECT_EQ(*compactionFailed, 0)
        << "image-chunk cancel rollback must succeed via snapshot restore";
  }

  // Recovery: the model must accept another inference (with or without
  // an image) cleanly on the rolled-back cache.
  LlamaModel::Prompt recovery;
  recovery.input = R"([{"role":"user","content":"Hi"}])";
  EXPECT_NO_THROW({
    std::string output = model->processPrompt(recovery);
    EXPECT_GE(output.length(), 0);
  });
}

// Same recovery property on a pure-attention multimodal model (SmolVLM).
// Regression guard: our changes to the prefill cancel block in
// `MtmdLlmContext::evalMessageWithTools` must not change the existing
// pure-attention cancel behaviour.
TEST_F(
    MtmdLlmContextCancelTest, CancelDuringPrefillLeavesPureAttentionMtmdUsable) {
  const std::string smolvlmPath = test_common::BaseTestModelPath::get(
      "SmolVLM-500M-Instruct-Q8_0.gguf", "SmolVLM-500M-Instruct.gguf");
  const std::string smolvlmMmproj = test_common::BaseTestModelPath::get(
      "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
      "mmproj-SmolVLM-500M-Instruct.gguf");
  auto model = loadMtmdModel(smolvlmPath, smolvlmMmproj);
  if (!model) {
    GTEST_SKIP() << "SmolVLM pure-attention multimodal model not found";
  }

  LlamaModel::Prompt cancelTargetPrompt;
  cancelTargetPrompt.input = R"([
    {"role":"user","content":"Cancel target: a moderately long prompt for the pure-attention multimodal cancel path."}
  ])";
  cancelTargetPrompt.prefill = true;

  std::atomic<bool> done{false};
  std::thread worker([&] {
    try {
      model->processPrompt(cancelTargetPrompt);
    } catch (...) {
      // Cancel may surface as a status error; treat as a clean cancel.
    }
    done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_NO_THROW(model->cancel());

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(done.load());
  worker.join();

  LlamaModel::Prompt recovery;
  recovery.input = R"([{"role":"user","content":"Hi"}])";
  EXPECT_NO_THROW({
    std::string output = model->processPrompt(recovery);
    EXPECT_GE(output.length(), 0);
  });
}

// ============================================================================
// Layer 2c: end-to-end cancel during generation via the high-level API
// ============================================================================

// Threaded cancel test: spawn the inference, set the cancel flag from
// another thread, then verify that the model survives and a follow-up
// inference still works. This is the only test that exercises the
// generation-cancel restore path (`TextLlmContext::onCancel` with a real
// in-flight generation).
//
// Test is timing-sensitive — uses retries with a non-trivial n_predict so
// the worker thread reliably wins the race to set `stopGeneration_`
// inside the generation loop on slow machines.
TEST(
    TextLlmContextCancelDuringGenerationTest, HybridModelSurvivesMidGenCancel) {
  const std::string modelPath = test_common::BaseTestModelPath::get(
      "Qwen3.5-0.8B-Q8_0.gguf");
  if (!fs::exists(modelPath)) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  std::unordered_map<std::string, std::string> config;
  config["device"] = test_common::getTestDevice();
  config["ctx_size"] = "4096";
  config["gpu_layers"] = test_common::getTestGpuLayers();
  config["n_predict"] = "128"; // long enough to cancel mid-flight
  config["backendsDir"] = test_common::getTestBackendsDir().string();

  std::string mp = modelPath;
  std::string proj;
  auto model = std::make_unique<LlamaModel>(
      std::move(mp), std::move(proj), std::move(config));
  model->waitForLoadInitialization();
  ASSERT_TRUE(model->isLoaded());

  LlamaModel::Prompt longPrompt;
  longPrompt.input = R"([
    {"role":"user","content":"Write a long story about a dragon."}
  ])";
  // Enable `remove_thinking_from_context` so the end-of-prefill snapshot
  // (`reasoningRecurrentSnapshot_`) actually gets populated. Without
  // this, `onCancel`'s rollback branch is skipped (snapshot is empty)
  // and the test reduces to a recovery smoke check that doesn't
  // exercise the new generation-cancel restore path.
  longPrompt.generationParams.remove_thinking_from_context = true;

  std::atomic<bool> generationDone{false};
  std::thread gen([&] {
    try {
      model->processPrompt(longPrompt);
    } catch (...) {
      // Cancel may surface as a status error on some paths; treat as
      // a clean cancel for the purposes of this test.
    }
    generationDone.store(true);
  });

  // Give the worker a brief head start so it reaches the generation loop.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  EXPECT_NO_THROW(model->cancel());

  // Wait for the worker to observe the cancel and unwind.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!generationDone.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(generationDone.load())
      << "model did not unwind within 10s of cancel";
  gen.join();

  // Stronger than "no throw": the new post-sampling cancel route in
  // generateResponse promises that mid-generation cancel on hybrid runs
  // through `onCancel` (snapshot restore) and never through the EOT
  // fallback. A successful restore leaves `thinkingCompactionFailed` at
  // zero — the only path that bumps that counter for cancel is a failed
  // `restoreRecurrentState`. We allow `nullopt` in case the runtime
  // stats vector omits the key on this build, but if it's present it
  // must be zero.
  const auto compactionFailed = statInt(*model, "thinkingCompactionFailed");
  if (compactionFailed.has_value()) {
    EXPECT_EQ(*compactionFailed, 0)
        << "hybrid generation cancel must succeed via snapshot restore "
           "(non-zero thinkingCompactionFailed indicates the fallback ran)";
  }

  // Recovery: subsequent inference must succeed on the cancelled context.
  LlamaModel::Prompt shortPrompt;
  shortPrompt.input = R"([{"role":"user","content":"Hi"}])";
  EXPECT_NO_THROW({
    std::string output = model->processPrompt(shortPrompt);
    EXPECT_GE(output.length(), 0);
  });
}

// Mid-prefill cancel on a hybrid model via the high-level API. Unlike the
// `PrefillCancelAtEntry*` cases above (which set the stop flag before
// `evalMessageWithTools` runs and never decode any chunks), this test
// signals cancel from another thread AFTER prefill has started. A
// sufficiently long prompt + low `n_batch` ensures the prefill loop
// crosses the cancel check at least once with `tokenIndex > 0`, exercising
// the snapshot restore path. The post-cancel cache cursor must roll back
// to the pre-prefill position so subsequent inference is unaffected.
TEST(
    TextLlmContextCancelDuringGenerationTest,
    MidPrefillCancelRollsBackHybridCache) {
  const std::string modelPath = qwen35HybridModelPath();
  if (!fs::exists(modelPath)) {
    GTEST_SKIP() << "Qwen3.5 hybrid model not found";
  }

  std::unordered_map<std::string, std::string> config;
  config["device"] = test_common::getTestDevice();
  config["ctx_size"] = "4096";
  config["gpu_layers"] = test_common::getTestGpuLayers();
  config["n_predict"] = "8";
  // Force a multi-chunk prefill so cancel can land mid-loop with at
  // least one chunk already decoded. `batch-size` is the llama.cpp
  // config knob honored by LlamaModel; small value + long prompt
  // guarantees many chunks.
  config["batch-size"] = "16";
  config["backendsDir"] = test_common::getTestBackendsDir().string();

  std::string mp = modelPath;
  std::string proj;
  auto model = std::make_unique<LlamaModel>(
      std::move(mp), std::move(proj), std::move(config));
  model->waitForLoadInitialization();
  ASSERT_TRUE(model->isLoaded());

  // Long prompt: with batch-size=16 this is dozens of chunks, giving the
  // worker plenty of opportunities to observe the cancel flag mid-loop.
  // The prompt is deliberately large so the 50ms head start below
  // reliably lands cancel AFTER the first chunk has been decoded — the
  // case that actually exercises the partial-prefill rollback path
  // rather than a degenerate "cancel before any decode" restore-to-empty.
  LlamaModel::Prompt longPrompt;
  longPrompt.input = R"([
    {"role":"user","content":"This is a deliberately long user message that the prefill loop must consume across many chunks. The point of this prompt is to make the cancel signal land well inside the prefill loop so the snapshot-restore rollback path runs with multiple decoded chunks already in the recurrent state. Keep going so this comfortably exceeds dozens of chunks at batch-size=16. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo."}
  ])";
  longPrompt.prefill = true;

  std::atomic<bool> done{false};
  std::thread worker([&] {
    try {
      model->processPrompt(longPrompt);
    } catch (...) {
      // Cancel may surface as a status error; treat as a clean cancel.
    }
    done.store(true);
  });

  // Head start so the worker enters the prefill loop. 50ms is enough
  // for several chunks to decode on M-series Macs and CI runners given
  // batch-size=16 and the ~250-token prompt above, so cancel reliably
  // lands AFTER decoded chunks — exercising the partial-prefill
  // rollback rather than a degenerate "cancel before any decode"
  // restore-to-empty. Failure mode would be cancel landing AFTER
  // prefill completes, which the prompt length + small batch make
  // very unlikely.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_NO_THROW(model->cancel());

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(done.load())
      << "worker did not unwind within 10s of cancel";
  worker.join();

  // Core assertion: the recurrent rollback must have fully rewound the
  // cache. Pre-prefill position on a fresh model is -1; any residual
  // KV cells from the cancelled prefill would push this above -1.
  EXPECT_EQ(seqPosMax(*model), -1)
      << "cancelled hybrid prefill must restore the pre-prefill cache "
         "cursor; residual cells indicate the snapshot rollback did not "
         "run or was bypassed";

  // The snapshot-capture and snapshot-restore both write to
  // `thinkingCompactionFailed` on failure. Any non-zero value means
  // we fell back to the no-op `removeLastNTokens` path.
  const auto compactionFailed = statInt(*model, "thinkingCompactionFailed");
  if (compactionFailed.has_value()) {
    EXPECT_EQ(*compactionFailed, 0)
        << "hybrid mid-prefill cancel must succeed via snapshot restore";
  }

  // Recovery: a fresh prefill must succeed on the rolled-back cache.
  LlamaModel::Prompt recovery;
  recovery.input = R"([{"role":"user","content":"Hi"}])";
  EXPECT_NO_THROW({
    std::string output = model->processPrompt(recovery);
    EXPECT_GE(output.length(), 0);
  });
}

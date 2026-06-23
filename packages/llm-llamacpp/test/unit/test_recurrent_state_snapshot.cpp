#include <vector>

#include <gtest/gtest.h>
#include <llama.h>

#include "utils/RecurrentStateSnapshot.hpp"

using namespace qvac_lib_inference_addon_llama::utils;

// Pure-logic coverage of the snapshot helpers. End-to-end coverage that
// touches a real `llama_context` lives in the `reasoning.test.js` and
// `gemma4.test.js` integration suites — those exercise the snapshot +
// restore + replay path against actual hybrid / pure-attention models.

TEST(RecurrentStateSnapshotTest, EmptyByDefault) {
  RecurrentStateSnapshot snap;
  EXPECT_TRUE(snap.empty());
  EXPECT_EQ(snap.size(), 0u);
  EXPECT_TRUE(snap.data.empty());
  EXPECT_EQ(snap.nPast, 0);
}

TEST(RecurrentStateSnapshotTest, ClearWipesPayloadAndNPast) {
  RecurrentStateSnapshot snap;
  snap.data = {1, 2, 3};
  snap.nPast = 42;
  ASSERT_FALSE(snap.empty());
  snap.clear();
  EXPECT_TRUE(snap.empty());
  EXPECT_EQ(snap.size(), 0u);
  EXPECT_EQ(snap.nPast, 0);
}

TEST(RecurrentStateSnapshotTest, SnapshotOnNullCtxFails) {
  RecurrentStateSnapshot snap;
  snap.data.push_back(0xAB);
  snap.nPast = 7;
  EXPECT_FALSE(snapshotRecurrentState(
      nullptr, /*seqId=*/0, /*nPastAt=*/12, snap));
  // Failure path always clears the buffer so a stale payload cannot
  // be accidentally restored later.
  EXPECT_TRUE(snap.empty());
  EXPECT_EQ(snap.nPast, 0);
}

TEST(RecurrentStateSnapshotTest, RestoreOnNullCtxFails) {
  RecurrentStateSnapshot snap;
  snap.data.push_back(0xCD);
  EXPECT_FALSE(restoreRecurrentState(nullptr, /*seqId=*/0, snap));
}

TEST(RecurrentStateSnapshotTest, RestoreEmptySnapshotIsNoOpButRequiresCtxSafety) {
  RecurrentStateSnapshot snap;
  // Empty snapshot + null ctx still returns false (we never reach the
  // empty-shortcut path because the ctx check guards first); this is
  // the documented contract — programming errors are surfaced.
  EXPECT_FALSE(restoreRecurrentState(nullptr, /*seqId=*/0, snap));
}

TEST(RecurrentStateSnapshotTest, ReplayEmptyTokensIsNoOpEvenWithNullCtx) {
  std::vector<llama_token> empty;
  EXPECT_TRUE(replayTokensThroughDecoder(
      nullptr, /*seqId=*/0, empty, /*startPos=*/0));
}

TEST(RecurrentStateSnapshotTest, ReplayNonEmptyTokensWithNullCtxFails) {
  std::vector<llama_token> tokens = {1, 2, 3};
  EXPECT_FALSE(replayTokensThroughDecoder(
      nullptr, /*seqId=*/0, tokens, /*startPos=*/0));
}

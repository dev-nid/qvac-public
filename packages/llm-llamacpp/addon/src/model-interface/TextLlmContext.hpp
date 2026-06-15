#pragma once

#include <atomic>
#include <utility>
#include <vector>

#include <llama.h>

#include "../utils/ChatTemplateUtils.hpp"
#include "../utils/Qwen3ReasoningUtils.hpp"
#include "../utils/UTF8TokenBuffer.hpp"
#include "LlmContext.hpp"
#include "SequenceDriver.hpp"
#include "ToolsCompactController.hpp"
#include "common/common.h"
#include "inference-addon-cpp/Logger.hpp"

/// Concrete text-only LLM context. Implements both the legacy
/// `LlmContext` API (driven by the single-prompt path in `LlamaModel`)
/// and the per-sequence `SequenceDriver` API (driven by the
/// `ContinuousBatchScheduler`). The overlapping state-query methods
/// (`getNPast`, `getNSlides`, `validatePromptPolicy`) appear on both
/// bases; a single override below satisfies both vtables.
class TextLlmContext : public LlmContext, public SequenceDriver {
public:
  TextLlmContext(const TextLlmContext&) = delete;
  TextLlmContext& operator=(const TextLlmContext&) = delete;
  TextLlmContext(TextLlmContext&&) = delete;
  TextLlmContext& operator=(TextLlmContext&&) = delete;
  // Constructor
  TextLlmContext(
      common_params& commonParams, common_init_result_ptr llamaInit,
      ToolsCompactController& tools);
  TextLlmContext(
      const common_params& commonParams, const LlmModelContext& shared,
      ToolsCompactController& tools, llama_seq_id seqId,
      llama_pos perSeqCtxCeiling = -1);

  // Destructor
  ~TextLlmContext() override = default;

  /**
   * The eval message method. It evaluates the message and updates the context.
   *
   * @param chatMsgs - chat messages.
   * @param is_cache_loaded - whether the cache is loaded.
   * @param prefill - whether to only prefill context without generation setup.
   * @return - true if successful, false if inference is stopped.
   */
  bool evalMessage(
      const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
      bool prefill) override;

  /**
   * The eval message with tools method. It evaluates the message with tools and
   * updates the context.
   *
   * @param chatMsgs - chat messages.
   * @param tools - tools.
   * @param isCacheLoaded - whether the cache is loaded.
   * @param prefill - whether to only prefill context without generation setup.
   * @return - true if successful, false if inference is stopped.
   */
  bool evalMessageWithTools(
      const std::vector<common_chat_msg>& chatMsgs,
      const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
      bool prefill) override;

  /**
   * The generate response method. It generates the response token by token.
   *
   * @param output_callback - the output callback.
   * @return - true if successful, false if context overflow.
   */
  bool generateResponse(
      const std::function<void(const std::string&)>& outputCallback) override;

  std::function<void()>
  applyGenerationParams(const GenerationParams& overrides) override;

  /**
   * The stop method. It stops the model inference.
   */
  void stop() override;

  /**
   * The get context method. It returns the context.
   *
   * @return - the context.
   */
  llama_context* getCtx() override;

  /**
   * Access the underlying llama model pointer.
   */
  llama_model* getModel() override { return modelCtx_.model; }

  /**
   * Access the mutable common parameters associated with this context.
   */
  common_params& getParams() override { return params_; }

  /**
   * The get n_past method. It returns the n_past.
   *
   * @return - the n_past.
   */
  [[nodiscard]] llama_pos getNPast() const override;

  /**
   * The set n_past method. It sets the n_past.
   *
   * @param n_past - the n_past.
   */
  void setNPast(llama_pos nPast) override;

  /**
   * The get first msg tokens method. It returns the first msg tokens.
   *
   * @return - the first msg tokens.
   */
  [[nodiscard]] llama_pos getFirstMsgTokens() const override;

  /**
   * The set first msg tokens method. It sets the first msg tokens.
   *
   * @param first_msg_tokens - the first msg tokens.
   */
  void setFirstMsgTokens(llama_pos firstMsgTokens) override;
  /**
   * The set n_discarded method. It sets the n_discarded.
   *
   * @param nDiscarded - the number of tokens to discard.
   */
  void setNDiscarded(llama_pos nDiscarded) override;

  /**
   * The get n_discarded method. It returns the configured context-shift
   * discard budget. A value of 0 means context shifting is disabled.
   *
   * @return - the number of tokens to discard on overflow.
   */
  [[nodiscard]] llama_pos getNDiscarded() const;

  [[nodiscard]] int32_t getNSlides() const override;
  void resetNSlides() override;

  /**
   * Number of `<think>`/`<channel|>` reasoning blocks that were
   * compacted out of the KV cache during the most recent generation.
   * Tracked per-inference and surfaced through `RuntimeStats` so
   * callers can observe how much cache reuse the feature reclaimed.
   *
   * @return - the count, or 0 when the feature was disabled / no
   *           reasoning blocks were emitted.
   */
  [[nodiscard]] int32_t getThinkingBlockDiscards() const override;
  void resetThinkingBlockDiscards() override;

  /**
   * The reset state method. It resets the context.
   *
   * @param resetStats - whether to reset performance statistics
   */
  void resetState(bool resetStats) override;

  /**
   * Remove the last N tokens from the model context.
   * This decrements n_past and removes the tokens from the KV cache.
   *
   * @param count - the number of tokens to remove
   * @return the actual number of tokens removed (may be less than requested if
   * not enough tokens exist)
   */
  llama_pos removeLastNTokens(llama_pos count) override;

  std::vector<llama_token> preparePrefill(
      const std::vector<common_chat_msg>& chatMsgs,
      const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
      bool prefill) override;

  void
  onPrefillComplete(llama_pos currentPos, size_t prefillTokenCount) override;

  void syncPosition(llama_pos currentPos) override;

  SequenceStepResult onLogitsReady(
      int logitIdx, unsigned generatedAfterAccept,
      const std::function<void(const std::string&)>& outputCallback,
      LlamaBatch* inlineDecodeBatch = nullptr) override;

  void onSequenceEnd(
      const std::function<void(const std::string&)>& outputCallback) override;

  void onGenerationFinished(
      const std::function<void(const std::string&)>& outputCallback) override;

  void onCancel(
      const std::function<void(const std::string&)>& outputCallback) override;

  void validatePromptPolicy(
      const std::vector<common_chat_msg>& chatMsgs,
      const std::vector<common_chat_tool>& tools, const PromptLayout& layout,
      bool hasKvCacheContext) const override;

  [[nodiscard]] bool loadCache(
      const std::string& cacheKey, llama_pos configuredNDiscarded) override;
  void saveCache(const std::string& cacheKey) const override;

private:
  /// Hook fired exactly once per slot, immediately before the policy
  /// flushes its UTF-8 buffer at end-of-generation. Internal helper for
  /// `onGenerationFinished`.
  void onGenerationCompletePolicy(std::string_view assistantOutput);

  /**
   * The check antiprompt method. It checks the antiprompt.
   *
   * @return - true if the antiprompt is found, false otherwise.
   */
  bool checkAntiprompt();

  /**
   * The Tokenize chat method. It tokenizes the chat.
   *
   * @param chatMsgs - chat messages.
   * @param inputTokens - output tokens.
   * @param isCacheLoaded - whether the cache is loaded.
   */
  void tokenizeChat(
      const std::vector<common_chat_msg>& chatMsgs,
      const std::vector<common_chat_tool>& tools,
      std::vector<llama_token>& inputTokens, bool isCacheLoaded);

  /// Replace an EOS token sampled while the model is still inside its
  /// reasoning channel with the model's close-tag token and inject the
  /// trailing newlines that the chat template expects.
  ///
  /// Only fires when the active model's reasoning close marker is a
  /// single token (i.e. `reasoningState_.cached_close_tag_token` is
  /// set). Multi-token close markers (e.g. Gemma 4's `<channel|>`)
  /// fall through to the regular EOS path — the model is responsible
  /// for closing its own channel in those cases.
  bool handleReasoningEOS(
      llama_token& tokenId, std::string& tokenStr, llama_batch& batch,
      llama_pos& nPast,
      const std::function<void(const std::string&)>& outputCallback);

  void flushPendingUtf8ToCallback(
      const std::function<void(const std::string&)>& outputCallback);
  void emitOutputPiece(
      const std::function<void(const std::string&)>& outputCallback,
      const std::string& text);
  void initializeCommonState();
  void initializeOwnedThreadpools();
  [[nodiscard]] llama_pos ctxCeiling() const;
  /// Slide the context window if the next token would not fit. Returns
  /// the number of tokens discarded (0 when no slide happened).
  llama_pos applyContextDiscard();
  void handleStopRequestAndAddEot(LlamaBatch& batch);

  /// Records the start position of a reasoning block that the chat
  /// template force-opened in the assistant prefix (Qwen3 / DeepSeek-R1
  /// style `<think>\n` suffix). Called once at the start of
  /// `generateResponse` when `thinkingForcedOpen_` is true. The end is
  /// filled in later by `capturePendingThinkClose` once the model
  /// closes the channel.
  void captureForcedOpenThinkSpanStart();

  /// Records the start position of a model-emitted reasoning block.
  /// Called from `onLogitsReady` on the buffer transition from
  /// "outside reasoning" to "inside reasoning".
  void captureModelOpenThinkSpanStart();

  /// Materialises a deferred close-position capture. Called at the
  /// top of `onLogitsReady` so the close marker emitted by the
  /// previous iteration has been committed to the KV cache before we
  /// read `nPast_`.
  void capturePendingThinkClose();

  /// Drops every completed thinking-block span recorded during this
  /// generation from the KV cache via `compactKvRange`. Called from
  /// `onGenerationFinished` and `onCancel` (the latter still cleans
  /// up any complete spans recorded before the cancel arrived).
  void compactThinkSpans();

  ToolsCompactController& tools_;
  common_init_result_ptr llamaInit_;
  LlmModelContext modelCtx_;
  CommonSamplerPtr smpl_;

  common_params params_;
  common_chat_templates_ptr tmpls_;
  std::vector<llama_token> antipromptTokens_;
  std::vector<llama_token> forcedTokens_;

  llama_pos nPast_ = 0;
  llama_pos nDiscarded_ = 0;
  llama_pos firstMsgTokens_ = 0;
  llama_pos perSeqCtxCeiling_ = -1;
  int32_t nSlides_ = 0;
  /// Number of `<think>` blocks compacted out of the KV cache during
  /// the current inference. Surfaced through `RuntimeStats`. Reset
  /// alongside `nSlides_` at the start of each inference.
  int32_t thinkingBlockDiscards_ = 0;
  bool pendingBatchFirstMsg_ = false;
  bool generationStarted_ = false;
  std::string assistantOutput_;
  ThreadPoolPtr threadpool_;
  ThreadPoolPtr threadpoolBatch_;

  // UTF-8 token buffer for handling incomplete emoji sequences
  qvac_lib_inference_addon_llama::UTF8TokenBuffer utf8Buffer_;

  // Reasoning detection state. `tags` are configured at construction
  // from `selectReasoningTagsForModel`; left empty for models without a
  // built-in reasoning channel.
  qvac_lib_inference_addon_llama::utils::ReasoningState reasoningState_;

  // True iff the active model exposes a recognised reasoning channel
  // (Qwen3 family, Gemma 4, ...). Checked once at load time and used as
  // a cheap gate on the per-token detection path.
  bool reasoningEnabled_ = false;

  // GPT-OSS Harmony: <|call|> is a frame delimiter, not a stop signal
  bool isHarmonyModel_ = false;
  llama_token harmonyCallToken_ = LLAMA_TOKEN_NULL;

  // Force-opens the reasoning channel in the prompt suffix to prepend the
  // matching "<think>\n" opener to the visible stream so consumers see balanced
  // tags.
  bool thinkingForcedOpen_ = false;

  /// Per-request toggle for the post-generation thinking-block KV
  /// cache compaction. Defaults to true. Flipped temporarily by
  /// `applyGenerationParams` when `generationParams.remove_thinking_
  /// from_context` is supplied; restored from the saved value when the
  /// returned restore lambda runs at end-of-request.
  bool removeThinkingFromContext_ = true;

  /// Inclusive-start / exclusive-end KV positions of each reasoning
  /// block emitted during the current generation. `end == -1` marks an
  /// open span (close marker not yet observed). Spans are accumulated
  /// in model order; `compactThinkSpans` walks them in reverse so
  /// earlier spans' positions stay valid as later spans are removed.
  std::vector<std::pair<llama_pos, llama_pos>> thinkSpans_;

  /// Set when `updateReasoningBuffer` (or the synthetic-close batch
  /// arm) transitions out of reasoning. The close-position capture is
  /// deferred to the top of the next `onLogitsReady` so the close
  /// marker token is already in the KV cache by the time we read
  /// `nPast_`.
  bool pendingThinkCloseCapture_ = false;

  std::atomic<bool> stopGeneration_ = false;
};

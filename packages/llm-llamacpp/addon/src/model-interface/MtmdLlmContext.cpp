#include "MtmdLlmContext.hpp"

#include <algorithm>
#include <cassert>

#include <common/log.h>
#include <inference-addon-cpp/Errors.hpp>
#include <llama/mtmd/mtmd-helper.h>
#include <llama/mtmd/mtmd.h>

#include "ContextSlider.hpp"
#include "GenerationParamsApply.hpp"
#include "addon/LlmErrors.hpp"
#include "inference-addon-cpp/Logger.hpp"
#include "utils/ChatTemplateUtils.hpp"
#include "utils/LoggingMacros.hpp"
#include "utils/RecurrentStateSnapshot.hpp"
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(readability-function-cognitive-complexity)

using namespace qvac_lib_inference_addon_llama::errors;
using namespace qvac_lib_inference_addon_cpp::logger;
using namespace qvac_lib_inference_addon_llama::utils;

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
MtmdLlmContext::MtmdLlmContext(
    common_params& commonParams, common_init_result_ptr llamaInit,
    ToolsCompactController& tools)
    : tools_(tools), llamaInit_(std::move(llamaInit)), params_(commonParams) {
  modelCtx_.model = llamaInit_->model();
  modelCtx_.lctx = llamaInit_->context();

  if (modelCtx_.model == nullptr) {
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(UnableToLoadModel),
        "Failed to initialize model.");
  }

  if (modelCtx_.lctx == nullptr) {
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(UnableToLoadModel),
        "Failed to initialize context");
  }

  modelCtx_.vocab = llama_model_get_vocab(modelCtx_.model);

  std::string chatTemplate =
      getChatTemplate(modelCtx_.model, params_, tools_.enabled());
  tmpls_ = common_chat_templates_init(modelCtx_.model, chatTemplate);

  smpl_.reset(common_sampler_init(modelCtx_.model, params_.sampling));
  if (!smpl_) {
    std::string errorMsg = string_format(
        "[MtmdLlm] %s: failed to initialize sampling subsystem\n", __func__);
    throw qvac_errors::StatusError(
        ADDON_ID, toString(UnableToCreateSamplingSystem), errorMsg);
  }

  if ((llama_model_chat_template(modelCtx_.model, nullptr) == nullptr) &&
      params_.chat_template.empty()) {
    QLOG_IF(
        Priority::ERROR,
        string_format(
            "[MtmdLlm] %s: Model does not have chat template\n", __func__));
    QLOG_IF(
        Priority::ERROR,
        "[MtmdLlm]   For old llava models, you may need to use "
        "'--chat-template "
        "vicuna'\n");
    QLOG_IF(
        Priority::ERROR,
        "[MtmdLlm]   For MobileVLM models, use '--chat-template deepseek'\n");
    QLOG_IF(
        Priority::ERROR,
        "[MtmdLlm]   For Mistral Small 3.1, use '--chat-template "
        "mistral-v7'\n");
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(
            qvac_errors::general_error::InvalidArgument),
        "Model does not have chat template");
  }

  initVisionContext();

  // antiprompt init
  for (const std::string& antiprompt : params_.antiprompt) {
    auto ids = ::common_tokenize(modelCtx_.lctx, antiprompt, false, true);
    if (ids.size() == 1) {
      antipromptTokens_.push_back(ids[0]);
    }
  }

  // load antiprompt tokens for legacy templates
  if (params_.chat_template == "vicuna") {
    auto tempTokens =
        common_tokenize(modelCtx_.lctx, "ASSISTANT:", false, true);
    antipromptTokens_.insert(
        antipromptTokens_.end(), tempTokens.begin(), tempTokens.end());
  } else if (params_.chat_template == "deepseek") {
    auto tempTokens = common_tokenize(modelCtx_.lctx, "###", false, true);
    antipromptTokens_.insert(
        antipromptTokens_.end(), tempTokens.begin(), tempTokens.end());
  }

  isHarmonyModel_ =
      qvac_lib_inference_addon_llama::utils::isHarmonyModel(modelCtx_.model);
  if (isHarmonyModel_) {
    harmonyCallToken_ =
        qvac_lib_inference_addon_llama::utils::getHarmonyCallToken(
            modelCtx_.lctx);
    if (harmonyCallToken_ == LLAMA_TOKEN_NULL) {
      isHarmonyModel_ = false;
    }
  }
  QLOG_IF(
      Priority::DEBUG,
      string_format(
          "[MtmdLlm] Harmony detection: isHarmony=%d callToken=%d\n",
          isHarmonyModel_,
          harmonyCallToken_));

  // Recurrent-memory detection: mirrors TextLlmContext. Covers fully-
  // recurrent (Mamba, RWKV) and hybrid SSM + attention families
  // (Qwen3.5 / Qwen3-Next / Jamba / Granite-Hybrid / ...) using
  // fabric's named contracts.
  hasRecurrentMemory_ =
      llama_model_is_recurrent(modelCtx_.model) ||
      llama_model_is_hybrid(modelCtx_.model);

  // EOS-inside-reasoning recovery is a Qwen3-specific workaround;
  // gate it on the explicit Qwen3-family predicate so non-Qwen
  // reasoning families (e.g. Gemma 4) don't inherit it. See
  // TextLlmContext for the same gate.
  {
    const std::optional<std::string> arch =
        qvac_lib_inference_addon_llama::utils::getModelArchitecture(
            modelCtx_.model);
    isQwen3ReasoningFamily_ =
        arch.has_value() &&
        qvac_lib_inference_addon_llama::utils::
            isQwen3ReasoningFamilyArchitecture(arch.value());
  }
  const std::optional<qvac_lib_inference_addon_llama::utils::ReasoningTags>
      reasoningTags =
          qvac_lib_inference_addon_llama::utils::selectReasoningTagsForModel(
              modelCtx_.model);
  if (reasoningTags.has_value()) {
    const bool reasoningInitOk =
        qvac_lib_inference_addon_llama::utils::initializeReasoningState(
            modelCtx_.lctx, reasoningState_, *reasoningTags);
    if (reasoningInitOk) {
      reasoningEnabled_ = true;
    } else {
      QLOG_IF(
          Priority::WARNING,
          string_format(
              "[MtmdLlm] reasoning detection disabled: first piece of open "
              "marker '%s' is not a special token under this vocab\n",
              reasoningTags->open.c_str()));
    }
  }
}

void MtmdLlmContext::initVisionContext() {
  const char* clipPath = params_.mmproj.path.c_str();
  mtmd_context_params mparams = mtmd_context_params_default();
  mparams.use_gpu = params_.mmproj_use_gpu;
  mparams.backend_device =
      params_.mmproj_backend.empty() ? nullptr : params_.mmproj_backend.c_str();
  mparams.print_timings = true;
  mparams.n_threads = params_.cpuparams.n_threads;
  ctxVision_.reset(mtmd_init_from_file(clipPath, modelCtx_.model, mparams));
  if (ctxVision_.get() == nullptr) {
    std::string errorMsg = string_format(
        "[MtmdLlm] Failed to load vision model from %s\n", clipPath);
    throw qvac_errors::StatusError(
        ADDON_ID, toString(UnableToLoadModel), errorMsg);
  }
}

bool MtmdLlmContext::checkAntiprompt() {
  if (!params_.antiprompt.empty()) {
    constexpr int kNPrev = 32;
    std::string lastOutput =
        common_sampler_prev_str(smpl_.get(), modelCtx_.lctx, kNPrev);

    // Check if each of the reverse prompts appears anywhere in the recent
    // output. We search the full kNPrev-token window because a single token
    // can decode to many characters, and a short antiprompt like "\n" may
    // appear at the start of such a token, far from the string's tail.
    // Matching is case-insensitive so callers don't have to list every
    // casing variant the model might emit.
    std::string lastOutputLower = lastOutput;
    std::transform(
        lastOutputLower.begin(),
        lastOutputLower.end(),
        lastOutputLower.begin(),
        [](unsigned char c) { return std::tolower(c); });
    for (const std::string& antiprompt : params_.antiprompt) {
      std::string antipromptLower = antiprompt;
      std::transform(
          antipromptLower.begin(),
          antipromptLower.end(),
          antipromptLower.begin(),
          [](unsigned char c) { return std::tolower(c); });
      if (lastOutputLower.find(antipromptLower) != std::string::npos) {
        return true;
      }
    }

    // check for reverse prompt using special tokens
    llama_token lastToken = common_sampler_last(smpl_.get());
    for (auto token : antipromptTokens_) {
      if (token == lastToken) {
        return true;
      }
    }
  }
  return false;
}

void MtmdLlmContext::tokenizeChat(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, mtmd::input_chunks& chunks,
    bool isCacheLoaded) {
  if (chatMsgs.empty()) {
    std::string errorMsg =
        string_format("[MtmdLlm] %s: no chat messages provided\n", __func__);
    throw qvac_errors::StatusError(ADDON_ID, toString(EmptyPrompt), errorMsg);
  }

  common_chat_templates_inputs inputs;
  std::string formattedChat;

  bool isLastMessageFromUser = false;
  bool addSpecial = false;

  if (current_.pos == 0 && !isCacheLoaded) {
    tools_.reset();
    const auto& lastRole = chatMsgs.back().role;
    isLastMessageFromUser = lastRole == "user" || lastRole == "tool";
    addSpecial = true;
  } else if (current_.pos > 0) {
    isLastMessageFromUser =
        chatMsgs.back().role == "user" || chatMsgs.back().role == "tool";
    common_sampler_reset(smpl_.get());
    addSpecial = false;
  }

  inputs.use_jinja = params_.use_jinja;
  inputs.enable_thinking = params_.reasoning_budget != 0;
  inputs.messages = chatMsgs;
  inputs.add_generation_prompt = isLastMessageFromUser;

  if (!tools.empty()) {
    inputs.tools = tools;
  }
  formattedChat = getPrompt(tmpls_.get(), inputs, &thinkingForcedOpen_);

  if (formattedChat.empty()) {
    std::string errorMsg = string_format(
        "[MtmdLlm] %s: formatted chat prompt is empty\n", __func__);
    throw qvac_errors::StatusError(ADDON_ID, toString(EmptyPrompt), errorMsg);
  }

  QLOG_IF(
      Priority::DEBUG,
      string_format("[MtmdLlm] formatted prompt: %s\n", formattedChat.c_str()));

  mtmd_input_text text;
  text.text = formattedChat.c_str();
  text.add_special = addSpecial;
  text.parse_special = true;

  auto bitmapsCPtr = bitmaps_.c_ptr();
  int32_t res = mtmd_tokenize(
      ctxVision_.get(),
      chunks.ptr.get(), // output
      &text,            // text
      bitmapsCPtr.data(),
      bitmapsCPtr.size());
  if (res != 0) {
    resetMedia();
    std::string errorMsg = string_format(
        "[MtmdLlm] %s: Unable to tokenize prompt, res = %d\n", __func__, res);
    throw qvac_errors::StatusError(ADDON_ID, toString(EncoderFailed), errorMsg);
  }

  if (tools_.enabled() && !tools.empty()) {
    inputs.tools = {};
    inputs.add_generation_prompt = false;
    inputs.use_jinja = params_.use_jinja;
    inputs.enable_thinking = params_.reasoning_budget != 0;
    auto promptNoTools = getPrompt(tmpls_.get(), inputs);

    if (!promptNoTools.empty()) {
      mtmd_input_text textNoTools;
      textNoTools.text = promptNoTools.c_str();
      textNoTools.add_special = addSpecial;
      textNoTools.parse_special = true;

      mtmd::input_chunks chunksNoTools(mtmd_input_chunks_init());
      int32_t resNoTools = mtmd_tokenize(
          ctxVision_.get(),
          chunksNoTools.ptr.get(),
          &textNoTools,
          bitmapsCPtr.data(),
          bitmapsCPtr.size());

      if (resNoTools == 0) {
        tools_.onTokenize(
            mtmd_helper_get_n_tokens(chunks.ptr.get()),
            mtmd_helper_get_n_tokens(chunksNoTools.ptr.get()));
      }
    }
  } else {
    tools_.onTokenize(mtmd_helper_get_n_tokens(chunks.ptr.get()), 0);
  }

  resetMedia();
}

bool MtmdLlmContext::evalMessage(
    const std::vector<common_chat_msg>& chatMsgs, bool isCacheLoaded,
    bool prefill) {
  return evalMessageWithTools(chatMsgs, {}, isCacheLoaded, prefill);
}

bool MtmdLlmContext::evalMessageWithTools(
    const std::vector<common_chat_msg>& chatMsgs,
    const std::vector<common_chat_tool>& tools, bool isCacheLoaded,
    bool prefill) {
  mtmd::input_chunks chunks(mtmd_input_chunks_init());

  tokenizeChat(chatMsgs, tools, chunks, isCacheLoaded);

  const bool isFirstMsg = (current_.pos == 0);

  const mtmd_input_chunks* chunksPtr = chunks.ptr.get();

  const llama_pos nTokens =
      static_cast<llama_pos>(mtmd_helper_get_n_tokens(chunksPtr));
  const llama_pos nPositions = mtmd_helper_get_n_pos(chunksPtr);
  if (nTokens >= llama_n_ctx(modelCtx_.lctx) ||
      nPositions >= llama_n_ctx(modelCtx_.lctx)) {
    std::string errorMsg = string_format(
        "[MtmdLlm] context overflow at prefill step (%d tokens, %d positions, "
        "max %d)\n",
        nTokens,
        nPositions,
        llama_n_ctx(modelCtx_.lctx));
    throw qvac_errors::StatusError(
        ADDON_ID, toString(ContextOverflow), errorMsg);
  }
  if (current_.pos + nPositions >= llama_n_ctx(modelCtx_.lctx) ||
      current_.cacheTokens + nTokens >= llama_n_ctx(modelCtx_.lctx)) {
    auto outcome = trySlidePrefill(
        modelCtx_.lctx,
        seqId_,
        current_,
        protectedPrefix_,
        ContextUsage{nPositions, nTokens},
        nDiscarded_,
        tools_,
        defaultContextSliderOps());
    switch (outcome.kind) {
    case ContextSlideOutcome::Kind::Slid:
      current_.pos = outcome.newNPast;
      current_.cacheTokens -= outcome.discarded;
      ++nSlides_;
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "[MtmdLlm] Prefill step: discarded %d tokens after the first "
              "message\n",
              outcome.discarded));
      break;
    case ContextSlideOutcome::Kind::FullWipe:
      current_.pos = outcome.newNPast;
      current_.cacheTokens = current_.pos == protectedPrefix_.pos
                                 ? protectedPrefix_.cacheTokens
                                 : protectedPrefix_.cacheTokens + 1;
      ++nSlides_;
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "[MtmdLlm] Prefill step: wiped %d tokens after the first "
              "message\n",
              outcome.discarded));
      break;
    case ContextSlideOutcome::Kind::Overflow: {
      std::string errorMsg = string_format(
          "[MtmdLlm] context overflow at prefill step (%d tokens, max "
          "%d)\n",
          current_.cacheTokens + nTokens,
          llama_n_ctx(modelCtx_.lctx));
      throw qvac_errors::StatusError(
          ADDON_ID, toString(ContextOverflow), errorMsg);
    }
    case ContextSlideOutcome::Kind::MemoryOperationFailed: {
      std::string errorMsg = string_format(
          "[MtmdLlm] failed to slide context memory at prefill step "
          "(nPast=%d, cacheTokens=%d, append=%d, max=%d)\n",
          current_.pos,
          current_.cacheTokens,
          nTokens,
          llama_n_ctx(modelCtx_.lctx));
      throw qvac_errors::StatusError(
          ADDON_ID, toString(ContextSlideFailed), errorMsg);
    }
    case ContextSlideOutcome::Kind::NotNeeded:
      break;
    }
  }

  size_t nChunks = mtmd_input_chunks_size(chunksPtr);
  if (nChunks == 0) {
    const char* errorMsg = "[MtmdLlm] Unable to eval prompt\n";
    throw qvac_errors::StatusError(ADDON_ID, toString(EncoderFailed), errorMsg);
  }

  llama_pos nPastLocal = current_.pos;

  for (size_t i = 0; i < nChunks; i++) {
    bool chunkLogitsLast = (i == nChunks - 1 && !prefill);
    const auto* chunk = mtmd_input_chunks_get(chunksPtr, i);

    if (stopGeneration_.load()) {
      llama_pos totalDelta = nPastLocal - current_.pos;
      current_.pos = nPastLocal;
      removeLastNTokens(totalDelta);
      stopGeneration_.store(false);
      return false;
    }
    int32_t res = mtmd_helper_eval_chunk_single(
        ctxVision_.get(),
        modelCtx_.lctx,
        chunk,
        nPastLocal,
        0,
        params_.n_batch,
        chunkLogitsLast,
        &nPastLocal);
    if (res != 0) {
      std::string errorMsg =
          "[MtmdLlm] failed to eval chunk " + std::to_string(i);
      throw qvac_errors::StatusError(
          ADDON_ID, toString(EncoderFailed), errorMsg);
    }
  }
  current_.pos = nPastLocal;
  current_.cacheTokens += nTokens;

  if (isFirstMsg) {
    protectedPrefix_ = current_;
    const auto ctxSize = static_cast<llama_pos>(llama_n_ctx(modelCtx_.lctx));
    if (nDiscarded_ >= ctxSize - protectedPrefix_.pos) {
      nDiscarded_ = ctxSize - protectedPrefix_.pos - 1;
    }
  }
  tools_.onEvalComplete(current_.pos, nPositions);
  return true;
}

void MtmdLlmContext::flushPendingUtf8ToCallback(
    const std::function<void(const std::string&)>& outputCallback) {
  if (!outputCallback || !utf8Buffer_.hasPendingBytes()) {
    return;
  }
  std::string remaining = utf8Buffer_.flush();
  if (!remaining.empty()) {
    outputCallback(remaining);
  }
}

void MtmdLlmContext::applyContextDiscard() {
  constexpr llama_pos effectiveCtx = -1;
  auto outcome = trySlideGeneration(
      modelCtx_.lctx,
      seqId_,
      current_.pos,
      protectedPrefix_.pos,
      nDiscarded_,
      tools_,
      defaultContextSliderOps(),
      effectiveCtx,
      current_.cacheTokens);
  if (outcome.kind == ContextSlideOutcome::Kind::Slid) {
    current_.pos = outcome.newNPast;
    current_.cacheTokens -= outcome.discarded;
    ++nSlides_;
    // Recorded span positions are no longer valid after the shift;
    // drop them along with the recurrent snapshot (which targeted the
    // pre-slide SSM state) and the post-reasoning capture buffer.
    thinkSpan_.reset();
    pendingThinkCloseCapture_ = false;
    reasoningRecurrentSnapshot_.clear();
    postReasoningTokens_.clear();
    capturingPostReasoning_ = false;
    QLOG_IF(
        Priority::DEBUG,
        string_format(
            "[MtmdLlm] discarded %d tokens after the first message\n",
            outcome.discarded));
  } else if (outcome.kind == ContextSlideOutcome::Kind::MemoryOperationFailed) {
    std::string errorMsg = string_format(
        "[MtmdLlm] failed to slide context memory during generation "
        "(nPast=%d, nDiscarded=%d)\n",
        current_.pos,
        nDiscarded_);
    throw qvac_errors::StatusError(
        ADDON_ID, toString(ContextSlideFailed), errorMsg);
  }
}

void MtmdLlmContext::handleStopRequestAndAddEot(LlamaBatch& batch) {
  stopGeneration_.store(false);
  llama_token eot = llama_vocab_eot(modelCtx_.vocab);
  common_batch_add(
      *batch,
      eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(modelCtx_.vocab) : eot,
      current_.pos++,
      {seqId_},
      true);
  if (llama_decode(modelCtx_.lctx, *batch) != 0) {
    const char* errorMsg = "[MtmdLlm] failed to decode EOT token\n";
    throw qvac_errors::StatusError(
        ADDON_ID, toString(FailedToDecode), errorMsg);
  }
  ++current_.cacheTokens;
}

bool MtmdLlmContext::generateResponse(
    const std::function<void(const std::string&)>& outputCallback) {

  int nRemain = params_.n_predict;
  LlamaBatch batch(1, 0, 1); // batch for next token generation

  // Per-inference reset of reasoning detection state. Mirrors
  // TextLlmContext::onPrefillComplete so consecutive generations don't
  // inherit a stale `inside_reasoning` flag or span.
  reasoningState_.inside_reasoning = false;
  reasoningState_.recent_output_buffer.clear();
  thinkSpan_.reset();
  pendingThinkCloseCapture_ = false;
  reasoningRecurrentSnapshot_.clear();
  postReasoningTokens_.clear();
  capturingPostReasoning_ = false;

  if (thinkingForcedOpen_) {
    if (outputCallback) {
      outputCallback("<think>\n");
    }
    // Template force-opened the reasoning channel: the open marker
    // tokens are already in the KV cache from prefill; record their
    // span so `compactThinkSpan` can drop them at end-of-generation.
    if (reasoningEnabled_) {
      setOpenThinkSpan(
          current_.pos -
          static_cast<llama_pos>(reasoningState_.forcedOpenTokenCount));
      reasoningState_.inside_reasoning = true;
    }
  }

  if (stopGeneration_.load()) {
    stopGeneration_.store(false);
    flushPendingUtf8ToCallback(outputCallback);
    return true;
  }

  while (nRemain != 0) {
    if (stopGeneration_.load()) {
      stopGeneration_.store(false);
      flushPendingUtf8ToCallback(outputCallback);
      return true;
    }
    if ((current_.pos + 1 >
             static_cast<llama_pos>(llama_n_ctx(modelCtx_.lctx)) ||
         current_.cacheTokens + 1 >
             static_cast<llama_pos>(llama_n_ctx(modelCtx_.lctx))) &&
        nDiscarded_ == 0) {
      QLOG_IF(
          Priority::WARNING,
          string_format(
              "[MtmdLlm] generation overflow: context is full and nDiscarded "
              "is "
              "0 (nPast=%d, nCtx=%d, firstMsgTokens=%d, nPastBeforeTools=%d, "
              "toolsCompact=%s)\n",
              current_.pos,
              llama_n_ctx(modelCtx_.lctx),
              protectedPrefix_.pos,
              tools_.anchor(),
              tools_.enabled() ? "true" : "false"));
      return false;
    }
    applyContextDiscard();

    llama_token tokenId =
        common_sampler_sample(smpl_.get(), modelCtx_.lctx, -1);
    common_sampler_accept(smpl_.get(), tokenId, true);
    --nRemain;

    std::string tokenStr =
        common_token_to_piece(modelCtx_.lctx, tokenId, params_.special);
    if (outputCallback) {
      std::string completeChars = utf8Buffer_.addToken(tokenStr);
      if (!completeChars.empty()) {
        outputCallback(completeChars);
      }
    }

    // Record post-reasoning tokens for replay on hybrid / recurrent
    // models. `capturingPostReasoning_` is set by the prior loop
    // iteration's `capturePendingThinkClose()` after the close marker
    // is committed.
    recordPostReasoningTokenIfActive(tokenId);

    // Reasoning channel detection. `current_.pos` here reflects the
    // cache state BEFORE this token is committed (it's incremented in
    // common_batch_add below), so the open-marker math mirrors
    // TextLlmContext: the first marker piece is at
    // `current_.pos - (openTokenCount - 1)`.
    if (reasoningEnabled_) {
      const bool wasInside = reasoningState_.inside_reasoning;
      qvac_lib_inference_addon_llama::utils::updateReasoningBuffer(
          tokenStr, reasoningState_);
      const bool nowInside = reasoningState_.inside_reasoning;
      if (!wasInside && nowInside) {
        setOpenThinkSpan(
            current_.pos -
            static_cast<llama_pos>(reasoningState_.openTokenCount - 1));
      }
      if (wasInside && !nowInside) {
        // Defer end capture — the close-marker token has not yet been
        // committed to the cache.
        pendingThinkCloseCapture_ = true;
      }
    }

    bool isEos = llama_vocab_is_eog(modelCtx_.vocab, tokenId);

    if (isEos && isHarmonyModel_ && params_.use_jinja &&
        tokenId == harmonyCallToken_) {
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "[MtmdLlm] Harmony <|call|> stop: tokenId=%d\n", tokenId));
      if (outputCallback) {
        std::string callMarker =
            common_token_to_piece(modelCtx_.lctx, tokenId, true);
        if (!callMarker.empty()) {
          outputCallback(callMarker);
        }
      }
      flushPendingUtf8ToCallback(outputCallback);
      break;
    }

    // EOS sampled while still inside the reasoning channel: substitute
    // the cached close marker, decode it so the span end position gets
    // recorded, then exit. Mirrors TextLlmContext single-prompt EOS
    // handling. Without this, `compactThinkSpan()` would skip removal
    // because `thinkSpan_->second` stays unset.
    if (isEos && isQwen3ReasoningFamily_ && reasoningState_.inside_reasoning &&
        reasoningState_.cached_close_tag_token != LLAMA_TOKEN_NULL) {
      tokenId = reasoningState_.cached_close_tag_token;
      tokenStr =
          common_token_to_piece(modelCtx_.lctx, tokenId, params_.special);
      reasoningState_.inside_reasoning = false;
      pendingThinkCloseCapture_ = true;

      if (outputCallback) {
        std::string completeChars = utf8Buffer_.addToken(tokenStr);
        if (!completeChars.empty()) {
          outputCallback(completeChars);
        }
      }

      common_batch_clear(*batch);
      common_batch_add(*batch, tokenId, current_.pos++, {seqId_}, true);
      if (llama_decode(modelCtx_.lctx, *batch) != 0) {
        const char* errorMsg =
            "[MtmdLlm] failed to decode substituted reasoning close tag\n";
        throw qvac_errors::StatusError(
            ADDON_ID, toString(FailedToDecode), errorMsg);
      }
      ++current_.cacheTokens;
      capturePendingThinkClose();
      flushPendingUtf8ToCallback(outputCallback);
      break;
    }

    if (isEos || checkAntiprompt()) {
      flushPendingUtf8ToCallback(outputCallback);
      break;
    }

    common_batch_clear(*batch);
    if (stopGeneration_.load()) {
      handleStopRequestAndAddEot(batch);
      break;
    }
    common_batch_add(*batch, tokenId, current_.pos++, {seqId_}, true);

    // eval the token
    if (llama_decode(modelCtx_.lctx, *batch) != 0) {
      const char* errorMsg = "[MtmdLlm] failed to decode next token\n";
      throw qvac_errors::StatusError(
          ADDON_ID, toString(FailedToDecode), errorMsg);
    }
    ++current_.cacheTokens;
    // Close-marker token (if any was sampled this iteration) is now
    // committed; capture the span end.
    capturePendingThinkClose();
  }

  if (nRemain == 0) {
    flushPendingUtf8ToCallback(outputCallback);
  }
  // Drop the reasoning block from the KV cache if the caller opted
  // in and a `<think>...</think>` (or model-equivalent) was emitted.
  compactThinkSpan();
  return true;
}

std::function<void()>
MtmdLlmContext::applyGenerationParams(const GenerationParams& overrides) {
  // Hybrid / fully-recurrent models (Qwen3.5, Qwen3-Next, Jamba, ...)
  // are supported via the snapshot + replay path in `compactThinkSpan`;
  // no pre-flight rejection is needed here. Snapshot or replay failure
  // logs and increments `thinkingCompactionFailed_` rather than
  // throwing, so the answer for the current turn is still delivered.
  auto restoreSampler = applyGenerationParamsToContext(
      params_, smpl_, modelCtx_.model, overrides);

  const bool savedRemoveThinking = removeThinkingFromContext_;
  bool toggled = false;
  if (overrides.remove_thinking_from_context) {
    removeThinkingFromContext_ = *overrides.remove_thinking_from_context;
    toggled = true;
  }

  if (!toggled) {
    return restoreSampler;
  }

  return [this,
          restoreSampler = std::move(restoreSampler),
          savedRemoveThinking]() {
    restoreSampler();
    removeThinkingFromContext_ = savedRemoveThinking;
  };
}

void MtmdLlmContext::stop() { stopGeneration_.store(true); }

llama_context* MtmdLlmContext::getCtx() { return modelCtx_.lctx; }

llama_pos MtmdLlmContext::getNPast() const { return current_.pos; }

void MtmdLlmContext::setNPast(llama_pos nPast) { current_.pos = nPast; }

llama_pos MtmdLlmContext::getCacheTokens() const {
  return current_.cacheTokens;
}

void MtmdLlmContext::setCacheTokens(llama_pos cacheTokens) {
  current_.cacheTokens = cacheTokens;
}

llama_pos MtmdLlmContext::getFirstMsgTokens() const {
  return protectedPrefix_.pos;
}

void MtmdLlmContext::setFirstMsgTokens(llama_pos firstMsgTokens) {
  protectedPrefix_.pos = firstMsgTokens;
}

llama_pos MtmdLlmContext::getFirstMsgCacheTokens() const {
  return protectedPrefix_.cacheTokens;
}

void MtmdLlmContext::setFirstMsgCacheTokens(llama_pos firstMsgCacheTokens) {
  protectedPrefix_.cacheTokens = firstMsgCacheTokens;
}

void MtmdLlmContext::setNDiscarded(llama_pos nDiscarded) {
  this->nDiscarded_ = nDiscarded;
}

int32_t MtmdLlmContext::getNSlides() const { return nSlides_; }
void MtmdLlmContext::resetNSlides() { nSlides_ = 0; }

int32_t MtmdLlmContext::getThinkingBlockDiscards() const {
  return thinkingBlockDiscards_;
}
void MtmdLlmContext::resetThinkingBlockDiscards() {
  thinkingBlockDiscards_ = 0;
}

int32_t MtmdLlmContext::getThinkingCompactionFailed() const {
  return thinkingCompactionFailed_;
}
void MtmdLlmContext::resetThinkingCompactionFailed() {
  thinkingCompactionFailed_ = 0;
}

void MtmdLlmContext::setOpenThinkSpan(llama_pos start) {
  // `start < 0` only for degenerate templates whose entire rendered
  // prompt is the forced-open suffix; drop the span and leave the
  // tokens in cache.
  if (start < 0) {
    return;
  }
  // Single-block policy: only the first `<think>...</think>` is tracked.
  if (thinkSpan_.has_value()) {
    return;
  }
  // Hybrid / fully-recurrent models: capture a full-state snapshot of
  // the sequence (attention KV + recurrent state) so the entire seq
  // can be rebuilt at end-of-generation via `state_seq_set_data_ext`
  // without needing `seq_rm` (which the recurrent module rejects for
  // partial-tail erasures that include the final committed pos). See
  // TextLlmContext::setOpenThinkSpan for the full rationale.
  if (removeThinkingFromContext_ && hasRecurrentMemory_) {
    if (!snapshotRecurrentState(
            modelCtx_.lctx,
            seqId_,
            current_.pos,
            reasoningRecurrentSnapshot_)) {
      QLOG_IF(
          Priority::WARNING,
          string_format(
              "[MtmdLlm] thinking-block compaction skipped: failed to "
              "snapshot sequence state at open marker (start=%d, "
              "pos=%d, seqId=%d)\n",
              start,
              current_.pos,
              seqId_));
      reasoningRecurrentSnapshot_.clear();
      ++thinkingCompactionFailed_;
      return;
    }
  }
  thinkSpan_ = std::make_pair(start, static_cast<llama_pos>(-1));
}

void MtmdLlmContext::capturePendingThinkClose() {
  if (!pendingThinkCloseCapture_) {
    return;
  }
  pendingThinkCloseCapture_ = false;
  if (!removeThinkingFromContext_ || !thinkSpan_.has_value()) {
    return;
  }
  if (thinkSpan_->second < 0) {
    thinkSpan_->second = current_.pos;
  }
  capturingPostReasoning_ =
      hasRecurrentMemory_ && !reasoningRecurrentSnapshot_.empty();
}

void MtmdLlmContext::recordPostReasoningTokenIfActive(llama_token tokenId) {
  if (!capturingPostReasoning_ || tokenId == LLAMA_TOKEN_NULL) {
    return;
  }
  postReasoningTokens_.push_back(tokenId);
}

void MtmdLlmContext::compactThinkSpan() {
  struct ResetReasoningState {
    MtmdLlmContext* self;
    ~ResetReasoningState() {
      self->thinkSpan_.reset();
      self->reasoningRecurrentSnapshot_.clear();
      self->postReasoningTokens_.clear();
      self->capturingPostReasoning_ = false;
    }
  } reset{this};

  if (!removeThinkingFromContext_ || !thinkSpan_.has_value()) {
    return;
  }
  const llama_pos start = thinkSpan_->first;
  const llama_pos end = thinkSpan_->second;

  if (end < 0 || end <= start) {
    return;
  }
  if (end > current_.pos) {
    return;
  }

  // Pure-attention path: the SSM is uninvolved, the existing seq_rm +
  // seq_add primitive plus per-multimodal cacheTokens accounting is
  // sufficient.
  if (!hasRecurrentMemory_ || reasoningRecurrentSnapshot_.empty()) {
    const CompactRangeOutcome outcome =
        compactKvRange(modelCtx_.lctx, seqId_, start, end, current_.pos);
    if (outcome.kind == CompactRangeOutcome::Kind::Compacted) {
      current_.pos = outcome.newNPast;
      // Multimodal cacheTokens tracks image tokens separately from text
      // positions; the compacted range is text-only so cacheTokens drops
      // by the same number of discarded tokens.
      current_.cacheTokens -= outcome.discarded;
      if (start < protectedPrefix_.pos) {
        const llama_pos removedProtectedTokens =
            std::min(outcome.discarded, protectedPrefix_.pos - start);
        protectedPrefix_.pos = start;
        protectedPrefix_.cacheTokens -= removedProtectedTokens;
      }
      if (tools_.enabled()) {
        tools_.onSlide(outcome.discarded, start);
      }
      ++thinkingBlockDiscards_;
      QLOG_IF(
          Priority::DEBUG,
          string_format(
              "[MtmdLlm] thinking-block compaction: dropped %d tokens "
              "[%d, %d), pos=%d, firstMsgTokens=%d\n",
              outcome.discarded,
              start,
              end,
              current_.pos,
              protectedPrefix_.pos));
    } else if (
        outcome.kind == CompactRangeOutcome::Kind::MemoryOperationFailed) {
      QLOG_IF(
          Priority::WARNING,
          string_format(
              "[MtmdLlm] thinking-block compaction failed: seqRm rejected "
              "range [%d, %d) (pos=%d, seqId=%d)\n",
              start,
              end,
              current_.pos,
              seqId_));
      ++thinkingCompactionFailed_;
    }
    return;
  }

  // Recurrent / hybrid path. See TextLlmContext::compactThinkSpan for
  // the full rationale: full-state restore + replay (no `seq_rm`).
  // Image chunks earlier in the prefix are part of the snapshot and
  // are restored along with the text tokens.
  const llama_pos snapshotPos = reasoningRecurrentSnapshot_.nPast;
  const size_t maxPost = static_cast<size_t>(current_.pos - end);
  if (postReasoningTokens_.size() > maxPost) {
    postReasoningTokens_.resize(maxPost);
  }

  if (!restoreRecurrentState(
          modelCtx_.lctx, seqId_, reasoningRecurrentSnapshot_)) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "[MtmdLlm] thinking-block compaction failed: full-state "
            "restore underflowed (start=%d, end=%d, snapshotPos=%d, "
            "seqId=%d)\n",
            start,
            end,
            snapshotPos,
            seqId_));
    ++thinkingCompactionFailed_;
    current_.cacheTokens -= (current_.pos - snapshotPos);
    current_.pos = snapshotPos;
    return;
  }

  if (!replayTokensThroughDecoder(
          modelCtx_.lctx, seqId_, postReasoningTokens_, snapshotPos)) {
    QLOG_IF(
        Priority::WARNING,
        string_format(
            "[MtmdLlm] thinking-block compaction failed: post-reasoning "
            "replay rejected (snapshotPos=%d, replayCount=%zu, "
            "seqId=%d)\n",
            snapshotPos,
            postReasoningTokens_.size(),
            seqId_));
    ++thinkingCompactionFailed_;
    current_.cacheTokens -= (current_.pos - snapshotPos);
    current_.pos = snapshotPos;
    return;
  }

  const llama_pos replayed =
      static_cast<llama_pos>(postReasoningTokens_.size());
  const llama_pos newPos = snapshotPos + replayed;
  const llama_pos discarded = current_.pos - newPos;
  current_.pos = newPos;
  // The discarded slice is text-only (no image chunks inside the
  // reasoning span), so cacheTokens drops by the same count.
  current_.cacheTokens -= discarded;
  if (snapshotPos < protectedPrefix_.pos) {
    const llama_pos removedProtectedTokens =
        std::min(discarded, protectedPrefix_.pos - snapshotPos);
    protectedPrefix_.pos = snapshotPos;
    protectedPrefix_.cacheTokens -= removedProtectedTokens;
  }
  if (tools_.enabled()) {
    tools_.onSlide(discarded, snapshotPos);
  }
  ++thinkingBlockDiscards_;
  QLOG_IF(
      Priority::DEBUG,
      string_format(
          "[MtmdLlm] thinking-block compaction (recurrent): dropped %d "
          "tokens (span [%d, %d), kept [0, %d)), replayed %zu post-"
          "reasoning tokens, pos=%d, firstMsgTokens=%d\n",
          discarded,
          start,
          end,
          snapshotPos,
          postReasoningTokens_.size(),
          current_.pos,
          protectedPrefix_.pos));
}

void MtmdLlmContext::loadMedia(const std::vector<uint8_t>& media) {
  if (media.empty()) {
    resetMedia();
    const char* errorMsg = "[MtmdLlm] Media buffer is empty\n";
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(
            qvac_errors::general_error::InvalidArgument),
        errorMsg);
  }

  if (ctxVision_.get() == nullptr) {
    resetMedia();
    const char* errorMsg = "[MtmdLlm] Vision context is not initialized\n";
    throw qvac_errors::StatusError(
        ADDON_ID, toString(UnableToLoadModel), errorMsg);
  }

  mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
      ctxVision_.get(), media.data(), media.size()));
  if (!bmp.ptr) {
    resetMedia();
    const char* errorMsg =
        "[MtmdLlm] Failed to load media from memory buffer\n";
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(
            qvac_errors::general_error::InvalidArgument),
        errorMsg);
  }
  bitmaps_.entries.push_back(std::move(bmp));
}

void MtmdLlmContext::loadMedia(const std::string& fname) {
  if (fname.empty()) {
    resetMedia();
    const char* errorMsg = "[MtmdLlm] Filename is empty\n";
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(
            qvac_errors::general_error::InvalidArgument),
        errorMsg);
  }

  if (ctxVision_.get() == nullptr) {
    resetMedia();
    const char* errorMsg = "[MtmdLlm] Vision context is not initialized\n";
    throw qvac_errors::StatusError(
        ADDON_ID, toString(UnableToLoadModel), errorMsg);
  }

  mtmd::bitmap bmp(
      mtmd_helper_bitmap_init_from_file(ctxVision_.get(), fname.c_str()));
  if (!bmp.ptr) {
    resetMedia();
    std::string errorMsg = string_format(
        "[MtmdLlm] Failed to load media from file: %s\n", fname.c_str());
    throw qvac_errors::StatusError(
        ADDON_ID,
        qvac_errors::general_error::toString(
            qvac_errors::general_error::InvalidArgument),
        errorMsg);
  }
  bitmaps_.entries.push_back(std::move(bmp));
}

void MtmdLlmContext::resetState(bool resetStats) {

  tools_.reset();
  current_ = {};
  protectedPrefix_ = {};

  // On partial reset (resetStats=false), preserve nSlides_,
  // thinkingBlockDiscards_, and thinkingCompactionFailed_ so
  // runtimeStats() can read the per-inference values. On full reset
  // (resetStats=true), clear them along with perf stats.
  if (resetStats) {
    nSlides_ = 0;
    thinkingBlockDiscards_ = 0;
    thinkingCompactionFailed_ = 0;
  }

  thinkSpan_.reset();
  pendingThinkCloseCapture_ = false;
  reasoningRecurrentSnapshot_.clear();
  postReasoningTokens_.clear();
  capturingPostReasoning_ = false;

  // Clear UTF-8 buffer when resetting state
  utf8Buffer_.clear();

  clearSequenceMemory(modelCtx_.lctx);

  // Reset the performance metrics
  if (resetStats) {
    llama_perf_context_reset(modelCtx_.lctx);
  }

  // Reset sampler if available
  common_sampler_reset(smpl_.get());

  // Synchronize to ensure all operations are complete
  llama_synchronize(modelCtx_.lctx);
}

void MtmdLlmContext::resetMedia() { bitmaps_.entries.clear(); }

llama_pos MtmdLlmContext::removeLastNTokens(llama_pos count) {
  // Validate input
  if (count <= 0) {
    return 0;
  }

  // Calculate how many tokens we can actually remove
  llama_pos tokensToRemove = std::min(count, current_.pos);

  if (tokensToRemove == 0) {
    return 0;
  }

  clearSequenceMemory(modelCtx_.lctx, current_.pos - tokensToRemove, -1);

  // Decrement the token count by the number of tokens removed
  current_.pos -= tokensToRemove;
  current_.cacheTokens -= std::min(tokensToRemove, current_.cacheTokens);

  // Note: The sampler doesn't have an "undo" function, so we leave it as is.
  // The sampler maintains its own history, but the removed tokens won't affect
  // future sampling since they're no longer in the KV cache.

  return tokensToRemove;
}

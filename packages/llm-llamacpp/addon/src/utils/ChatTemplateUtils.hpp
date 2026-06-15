#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "Qwen3ReasoningUtils.hpp"
#include "common/chat.h"
#include "common/common.h"

// Forward declaration from llama.h
struct llama_model;
struct llama_context;

namespace qvac_lib_inference_addon_llama {
namespace utils {

bool isQwen3Model(const ::llama_model* model);
bool isHarmonyModel(const ::llama_model* model);
/// True when the model is a Gemma 4 variant. Detection combines the
/// GGUF `general.architecture` value (`"gemma4"` when llama.cpp has a
/// dedicated arch enum) and the `general.basename` metadata (covers
/// pre-arch-enum packs like bartowski's `google_gemma-4-*` GGUFs that
/// still report a generic Gemma architecture).
bool isGemma4Model(const ::llama_model* model);
llama_token getHarmonyCallToken(::llama_context* lctx);
std::optional<std::string> getModelArchitecture(const ::llama_model* model);
bool supportsToolsCompactForModelMetadata(
    const std::optional<std::string>& architecture);

/// Returns the reasoning channel open / close markers for a model
/// family that exposes a built-in reasoning channel.
///
/// Currently recognised families:
///   * Qwen3 family (incl. fine-tunes) -> `<think>` / `</think>`
///   * Gemma 4                         -> `<|channel>thought` / `<channel|>`
///
/// Returns `std::nullopt` for any other model. Callers should treat
/// that as "no reasoning channel" and skip detection / KV-compaction
/// rather than guessing tags (a wrong-guess produces silent KV cache
/// corruption, whereas missing detection is harmless).
///
/// Add new families by extending this function. Per-family token-count
/// metadata that depends on the active tokenizer is captured separately
/// in `ReasoningState` by `initializeReasoningState`.
std::optional<ReasoningTags>
selectReasoningTagsForModel(const ::llama_model* model);

/**
 * @brief Returns true when the GGUF metadata basename identifies a MedPsy
 * model. Exposed for unit testing without requiring a real ::llama_model.
 *
 * Comparison is case-insensitive against the literal "MedPsy"; an empty
 * basename returns false (callers should pass `value_or("")` from the
 * upstream `std::optional<std::string>` metadata accessor).
 */
bool isMedPsyBasename(std::string_view basename);

/**
 * @brief Returns true when the model's `general.basename` metadata identifies
 * it as a MedPsy model. MedPsy ships its own chat template embedded in the
 * GGUF, so callers should defer to it rather than substituting the hardcoded
 * Qwen3 templates.
 */
bool isMedPsyModel(const ::llama_model* model);

/**
 * @brief Returns true when `basename` (case-insensitive) contains one of
 * the Gemma 4 marker substrings ("gemma-4", "gemma 4", "gemma4"). Exposed
 * for unit testing without requiring a real ::llama_model. Empty input
 * returns false.
 */
bool isGemma4Basename(std::string_view basename);

std::optional<std::string> selectToolsCompactMarkerForModelMetadata(
    const std::optional<std::string>& architecture);

/**
 * @brief Gets the appropriate chat template for a model
 *
 * Resolution order:
 *   1. A non-empty `manualOverride` always wins.
 *   2. Models whose GGUF `general.basename` is "MedPsy" return an empty
 *      string so callers fall through to the embedded chat template, even
 *      when the architecture is reported as qwen3.
 *   3. Qwen3 models return either the tools-compact dynamic template or the
 *      fixed Qwen3 template based on the `toolsCompact` flag.
 *   4. All other models return an empty string.
 */
std::string getChatTemplateForModel(
    const ::llama_model* model, const std::string& manualOverride,
    bool toolsCompact);

/**
 * @brief Gets the chat template for a model, applying Qwen3 fixes if Jinja is
 * enabled
 */
std::string getChatTemplate(
    const ::llama_model* model, const common_params& params, bool toolsCompact);

/**
 * @brief Applies chat templates to generate a prompt, with fallback handling
 * for models that don't support tools.
 *
 * @p outThinkingForcedOpen (optional) receives the flag indicating that the
 *    template force-opened the reasoning channel in the prompt suffix.
 */
std::string getPrompt(
    const struct common_chat_templates* tmpls,
    struct common_chat_templates_inputs& inputs,
    bool* outThinkingForcedOpen = nullptr);

} // namespace utils
} // namespace qvac_lib_inference_addon_llama

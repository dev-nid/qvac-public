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
bool isGemma4Model(const ::llama_model* model);
llama_token getHarmonyCallToken(::llama_context* lctx);
std::optional<std::string> getModelArchitecture(const ::llama_model* model);
bool supportsToolsCompactForModelMetadata(
    const std::optional<std::string>& architecture);

// Reasoning channel markers for the model family, or std::nullopt
// when the family has no recognised channel. Extend the table here
// to add support for new families.
std::optional<ReasoningTags>
selectReasoningTagsForModel(const ::llama_model* model);

// Architecture-only variant. Covers families identifiable from
// `general.architecture` alone (Qwen3-family: `qwen3`, `qwen35`,
// `qwen35moe`, ...). Gemma 4 needs basename and is resolved in
// `selectReasoningTagsForModel`.
std::optional<ReasoningTags> selectReasoningTagsForArchitecture(
    const std::optional<std::string>& architecture);

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
 * @brief Returns true when `basename` (case-insensitive) contains a
 * Gemma 4 marker substring. Exposed for unit testing without requiring
 * a real ::llama_model.
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

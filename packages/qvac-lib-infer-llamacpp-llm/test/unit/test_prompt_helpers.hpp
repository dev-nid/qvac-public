#pragma once

#include <memory>
#include <optional>
#include <string>

#include "model-interface/LlamaModel.hpp"

namespace test_common {

inline std::string processPromptString(
    const std::unique_ptr<LlamaModel>& model, const std::string& input) {
  LlamaModel::Prompt prompt;
  prompt.input = input;
  return model->processPrompt(prompt);
}

inline std::string processPromptWithCacheOptions(
    const std::unique_ptr<LlamaModel>& model, const std::string& input,
    const std::string& cacheKey, bool resetCache = false,
    std::optional<std::string> persistTo = std::nullopt) {
  LlamaModel::Prompt prompt;
  prompt.input = input;
  prompt.cacheKey = cacheKey;
  prompt.resetCache = resetCache;
  prompt.persistTo = persistTo;
  return model->processPrompt(prompt);
}

} // namespace test_common

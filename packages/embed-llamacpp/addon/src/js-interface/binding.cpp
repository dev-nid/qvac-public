#include <bare.h>

#include "../addon/AddonJs.hpp"

// Forward declaration for the IdMapIndex (vector-index) binding registrar.
// The implementation lives in `vector-index-binding.cpp` and is deliberately
// kept in its own TU so it has no symbol dependency on BertModel /
// LlamaLazyInitializeBackend — required for the POC's lifecycle-isolation
// invariant (constructing IdMapIndex must not boot fabric's LLM backend).
namespace qvac_lib_inference_addon_embed::vector_index {
void registerBindings(js_env_t* env, js_value_t* exports);
}

js_value_t*
qvacLibInferLlamacppEmbedExports(js_env_t* env, js_value_t* exports) {

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define V(name, fn)                                                            \
  {                                                                            \
    js_value_t* val;                                                           \
    if (js_create_function(env, name, -1, fn, nullptr, &val) != 0) {           \
      return nullptr;                                                          \
    }                                                                          \
    if (js_set_named_property(env, exports, name, val) != 0) {                 \
      return nullptr;                                                          \
    }                                                                          \
  }

  V("createInstance", qvac_lib_inference_addon_embed::createInstance)
  V("runJob", qvac_lib_inference_addon_embed::runJob)

  V("loadWeights", qvac_lib_inference_addon_cpp::JsInterface::loadWeights)
  V("activate", qvac_lib_inference_addon_cpp::JsInterface::activate)
  V("cancel", qvac_lib_inference_addon_cpp::JsInterface::cancel)
  V("destroyInstance",
    qvac_lib_inference_addon_cpp::JsInterface::destroyInstance)
  V("setLogger", qvac_lib_inference_addon_cpp::JsInterface::setLogger)
  V("releaseLogger", qvac_lib_inference_addon_cpp::JsInterface::releaseLogger)
#undef V
// NOLINTEND(cppcoreguidelines-macro-usage)

  qvac_lib_inference_addon_embed::vector_index::registerBindings(env, exports);

  return exports;
}

BARE_MODULE("embed-llamacpp", qvacLibInferLlamacppEmbedExports)

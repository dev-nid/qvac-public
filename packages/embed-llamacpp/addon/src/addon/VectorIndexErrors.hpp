#pragma once
//
// Maps fabric vector-index C error codes to JS-throwable messages. Mirrors
// the layout of `BertErrors.hpp` but for the ANN index path. Kept in its own
// header so the binding can include it without dragging in any BertModel /
// LlamaLazyInitializeBackend symbols (lifecycle isolation requirement of
// the POC).

#include <ggml-vector-index.h>

#include <string>

namespace qvac_lib_infer_llamacpp_embed::vector_index_errors {

constexpr const char* ADDON_ID = "IdMapIndex";

inline std::string toString(int code) {
  switch (code) {
    case GGML_VEC_INDEX_OK:
      return "OK";
    case GGML_VEC_INDEX_E_INVALID_DIM:
      return "InvalidDim";
    case GGML_VEC_INDEX_E_INVALID_ARG:
      return "InvalidArgument";
    case GGML_VEC_INDEX_E_DUPLICATE:
      return "DuplicateId";
    case GGML_VEC_INDEX_E_IO:
      return "IOError";
    case GGML_VEC_INDEX_E_BAD_MAGIC:
      return "BadMagic";
    case GGML_VEC_INDEX_E_BAD_VERSION:
      return "BadVersion";
    case GGML_VEC_INDEX_E_OOM:
      return "OutOfMemory";
    case GGML_VEC_INDEX_E_INTERNAL:
      return "InternalError";
    default:
      return "UnknownError";
  }
}

} // namespace qvac_lib_infer_llamacpp_embed::vector_index_errors

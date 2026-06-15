// vector-index-binding.cpp
//
// N-API surface for the `IdMapIndex` JS class. Registers a small set of free
// functions on the embed-llamacpp addon's exports; the JS wrapper in
// `idMapIndex.js` ties them into a class shape.
//
// Lifecycle isolation: this binding deliberately depends ONLY on the
// VectorIndex C++ wrapper (which in turn depends only on fabric's
// ggml-vector-index C API). It never references BertModel,
// LlamaLazyInitializeBackend, or any other BERT-runtime symbol, so simply
// importing the addon does not boot fabric's LLM backend. The same .bare
// binary carries both class surfaces; the JS side decides which to construct.

#include <bare.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>

#include "../addon/VectorIndexErrors.hpp"
#include "../model-interface/VectorIndex.hpp"

namespace {

using qvac_lib_infer_llamacpp_embed::VectorIndex;
namespace verrors = qvac_lib_infer_llamacpp_embed::vector_index_errors;

// Finalizer: invoked by the JS engine when the external handle is GC'd.
// Tears down the native C handle via VectorIndex's RAII dtor.
void
finalize_vector_index(js_env_t* /*env*/, void* data, void* /*hint*/) {
  auto* idx = static_cast<VectorIndex*>(data);
  delete idx;
}

// Wrap an already-constructed VectorIndex into a JS external. Takes
// ownership of `idx`; the JS engine will delete it via finalize on GC.
js_value_t* wrap(js_env_t* env, VectorIndex* idx) {
  js_value_t* external = nullptr;
  if (js_create_external(env, idx, finalize_vector_index, nullptr, &external)
      != 0) {
    delete idx;
    js_throw_error(env, "InternalError", "failed to create external");
    return nullptr;
  }
  return external;
}

// Get a borrowed pointer out of a JS external handle. Throws and returns
// null on failure.
VectorIndex* unwrap(js_env_t* env, js_value_t* handle) {
  void* data = nullptr;
  if (js_get_value_external(env, handle, &data) != 0 || data == nullptr) {
    js_throw_error(env, "InvalidArgument", "expected IdMapIndex handle");
    return nullptr;
  }
  return static_cast<VectorIndex*>(data);
}

// Read a JS object property and parse as int32. On failure, throws and
// returns the provided default; caller should check pending exception via
// js_is_exception_pending or by returning nullptr from the function.
bool read_int_prop(
    js_env_t* env, js_value_t* obj, const char* name, int32_t* out) {
  js_value_t* val = nullptr;
  if (js_get_named_property(env, obj, name, &val) != 0) {
    return false;
  }
  return js_get_value_int32(env, val, out) == 0;
}

void throw_status(js_env_t* env, int code) {
  const std::string name = verrors::toString(code);
  js_throw_error(env, name.c_str(), name.c_str());
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// idx_create({ dim, bitWidth }) -> external handle
js_value_t* idx_create(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 1;
  js_value_t* argv[1] = { nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 1) {
    js_throw_type_error(env, "InvalidArgument", "expected { dim, bitWidth }");
    return nullptr;
  }
  int32_t dim = 0;
  int32_t bit_width = 32;
  if (!read_int_prop(env, argv[0], "dim", &dim)) {
    js_throw_type_error(env, "InvalidArgument", "missing or invalid `dim`");
    return nullptr;
  }
  // bitWidth optional; default to 32 if missing.
  (void) read_int_prop(env, argv[0], "bitWidth", &bit_width);

  VectorIndex* idx = nullptr;
  try {
    idx = new VectorIndex(dim, bit_width);
  } catch (const std::invalid_argument& e) {
    js_throw_error(env, "InvalidArgument", e.what());
    return nullptr;
  } catch (const std::bad_alloc&) {
    js_throw_error(env, "OutOfMemory", "allocation failure");
    return nullptr;
  }
  return wrap(env, idx);
}

// idx_load(path) -> external handle (throws on file errors).
js_value_t* idx_load(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 1;
  js_value_t* argv[1] = { nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 1) {
    js_throw_type_error(env, "InvalidArgument", "expected path string");
    return nullptr;
  }
  size_t len = 0;
  if (js_get_value_string_utf8(env, argv[0], nullptr, 0, &len) != 0) {
    js_throw_type_error(env, "InvalidArgument", "path must be a string");
    return nullptr;
  }
  std::string path(len, '\0');
  size_t copied = 0;
  if (js_get_value_string_utf8(
          env, argv[0],
          reinterpret_cast<utf8_t*>(path.data()), len + 1, &copied) != 0) {
    js_throw_error(env, "InternalError", "failed to read path string");
    return nullptr;
  }
  path.resize(copied);

  VectorIndex loaded = VectorIndex::load(path);
  if (!loaded.valid()) {
    js_throw_error(env, "IOError", "ggml_vec_index_load returned null");
    return nullptr;
  }
  // Move the wrapper onto the heap so we can hand JS an owning external.
  auto* heap = new VectorIndex(std::move(loaded));
  return wrap(env, heap);
}

// idx_add(handle, Float32Array vectors, BigUint64Array ids) -> undefined
js_value_t* idx_add(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 3;
  js_value_t* argv[3] = { nullptr, nullptr, nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 3) {
    js_throw_type_error(env, "InvalidArgument",
        "expected (handle, vectors:Float32Array, ids:BigUint64Array)");
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }

  js_typedarray_type_t vtype{};
  void* vdata = nullptr;
  size_t vlen = 0;
  if (js_get_typedarray_info(
          env, argv[1], &vtype, &vdata, &vlen, nullptr, nullptr) != 0
      || vtype != js_float32array) {
    js_throw_type_error(env, "InvalidArgument",
        "vectors must be a Float32Array");
    return nullptr;
  }

  js_typedarray_type_t itype{};
  void* idata = nullptr;
  size_t ilen = 0;
  if (js_get_typedarray_info(
          env, argv[2], &itype, &idata, &ilen, nullptr, nullptr) != 0
      || itype != js_biguint64array) {
    js_throw_type_error(env, "InvalidArgument",
        "ids must be a BigUint64Array");
    return nullptr;
  }

  const int dim = idx->dim();
  if (dim <= 0) {
    js_throw_error(env, "InternalError", "index has invalid dim");
    return nullptr;
  }
  if (vlen != ilen * static_cast<size_t>(dim)) {
    js_throw_range_error(env, "InvalidArgument",
        "vectors.length must equal ids.length * dim");
    return nullptr;
  }
  if (ilen > static_cast<size_t>(INT32_MAX)) {
    js_throw_range_error(env, "InvalidArgument", "too many vectors in batch");
    return nullptr;
  }

  const int rc = idx->add(
      static_cast<const float*>(vdata),
      static_cast<int>(ilen),
      static_cast<const uint64_t*>(idata));
  if (rc != 0) {
    throw_status(env, rc);
    return nullptr;
  }
  js_value_t* u = nullptr;
  js_get_undefined(env, &u);
  return u;
}

// idx_search(handle, Float32Array queries, int k)
//   -> { scores: Float32Array(m*k), ids: BigUint64Array(m*k), m, k }
js_value_t* idx_search(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 3;
  js_value_t* argv[3] = { nullptr, nullptr, nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 3) {
    js_throw_type_error(env, "InvalidArgument",
        "expected (handle, queries:Float32Array, k:number)");
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }

  js_typedarray_type_t qtype{};
  void* qdata = nullptr;
  size_t qlen = 0;
  if (js_get_typedarray_info(
          env, argv[1], &qtype, &qdata, &qlen, nullptr, nullptr) != 0
      || qtype != js_float32array) {
    js_throw_type_error(env, "InvalidArgument",
        "queries must be a Float32Array");
    return nullptr;
  }

  int32_t k = 0;
  if (js_get_value_int32(env, argv[2], &k) != 0 || k <= 0) {
    js_throw_type_error(env, "InvalidArgument", "k must be a positive int");
    return nullptr;
  }

  const int dim = idx->dim();
  if (dim <= 0 || qlen % static_cast<size_t>(dim) != 0) {
    js_throw_range_error(env, "InvalidArgument",
        "queries.length must be a multiple of dim");
    return nullptr;
  }
  const size_t m = qlen / static_cast<size_t>(dim);
  if (m > static_cast<size_t>(INT32_MAX)) {
    js_throw_range_error(env, "InvalidArgument", "too many queries");
    return nullptr;
  }

  // Allocate output ArrayBuffers; we'll hand them to JS via typed-array
  // views.
  const size_t total      = m * static_cast<size_t>(k);
  const size_t scores_b   = total * sizeof(float);
  const size_t ids_b      = total * sizeof(uint64_t);

  void* scores_data = nullptr;
  js_value_t* scores_ab = nullptr;
  if (js_create_arraybuffer(env, scores_b, &scores_data, &scores_ab) != 0) {
    js_throw_error(env, "OutOfMemory", "scores arraybuffer");
    return nullptr;
  }
  void* ids_data = nullptr;
  js_value_t* ids_ab = nullptr;
  if (js_create_arraybuffer(env, ids_b, &ids_data, &ids_ab) != 0) {
    js_throw_error(env, "OutOfMemory", "ids arraybuffer");
    return nullptr;
  }

  const int rc = idx->search(
      static_cast<const float*>(qdata),
      static_cast<int>(m),
      k,
      static_cast<float*>(scores_data),
      static_cast<uint64_t*>(ids_data));
  if (rc != 0) {
    throw_status(env, rc);
    return nullptr;
  }

  js_value_t* scores_ta = nullptr;
  if (js_create_typedarray(
          env, js_float32array, total, scores_ab, 0, &scores_ta) != 0) {
    js_throw_error(env, "InternalError", "create scores typedarray");
    return nullptr;
  }
  js_value_t* ids_ta = nullptr;
  if (js_create_typedarray(
          env, js_biguint64array, total, ids_ab, 0, &ids_ta) != 0) {
    js_throw_error(env, "InternalError", "create ids typedarray");
    return nullptr;
  }

  js_value_t* result = nullptr;
  if (js_create_object(env, &result) != 0) {
    js_throw_error(env, "InternalError", "create result object");
    return nullptr;
  }
  if (js_set_named_property(env, result, "scores", scores_ta) != 0
      || js_set_named_property(env, result, "ids", ids_ta) != 0) {
    js_throw_error(env, "InternalError", "set result fields");
    return nullptr;
  }
  js_value_t* m_val = nullptr;
  js_value_t* k_val = nullptr;
  js_create_uint32(env, static_cast<uint32_t>(m), &m_val);
  js_create_int32(env, k, &k_val);
  js_set_named_property(env, result, "m", m_val);
  js_set_named_property(env, result, "k", k_val);
  return result;
}

// idx_remove(handle, id:bigint) -> boolean
js_value_t* idx_remove(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 2;
  js_value_t* argv[2] = { nullptr, nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 2) {
    js_throw_type_error(env, "InvalidArgument",
        "expected (handle, id:bigint)");
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }

  uint64_t id = 0;
  bool lossless = false;
  if (js_get_value_bigint_uint64(env, argv[1], &id, &lossless) != 0
      || !lossless) {
    js_throw_type_error(env, "InvalidArgument",
        "id must be an unsigned BigInt fitting in 64 bits");
    return nullptr;
  }

  const int rc = idx->remove(id);
  if (rc < 0) {
    throw_status(env, rc);
    return nullptr;
  }
  js_value_t* result = nullptr;
  js_get_boolean(env, rc == 1, &result);
  return result;
}

// idx_contains(handle, id:bigint) -> boolean
js_value_t* idx_contains(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 2;
  js_value_t* argv[2] = { nullptr, nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 2) {
    js_throw_type_error(env, "InvalidArgument",
        "expected (handle, id:bigint)");
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }

  uint64_t id = 0;
  bool lossless = false;
  if (js_get_value_bigint_uint64(env, argv[1], &id, &lossless) != 0
      || !lossless) {
    js_throw_type_error(env, "InvalidArgument",
        "id must be an unsigned BigInt fitting in 64 bits");
    return nullptr;
  }
  js_value_t* result = nullptr;
  js_get_boolean(env, idx->contains(id), &result);
  return result;
}

// idx_prepare(handle) -> undefined (no-op in POC).
js_value_t* idx_prepare(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 1;
  js_value_t* argv[1] = { nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }
  idx->prepare();
  js_value_t* u = nullptr;
  js_get_undefined(env, &u);
  return u;
}

// idx_write(handle, path) -> undefined; throws on IO error.
js_value_t* idx_write(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 2;
  js_value_t* argv[2] = { nullptr, nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  if (argc < 2) {
    js_throw_type_error(env, "InvalidArgument", "expected (handle, path)");
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }

  size_t len = 0;
  if (js_get_value_string_utf8(env, argv[1], nullptr, 0, &len) != 0) {
    js_throw_type_error(env, "InvalidArgument", "path must be a string");
    return nullptr;
  }
  std::string path(len, '\0');
  size_t copied = 0;
  if (js_get_value_string_utf8(
          env, argv[1],
          reinterpret_cast<utf8_t*>(path.data()), len + 1, &copied) != 0) {
    js_throw_error(env, "InternalError", "failed to read path");
    return nullptr;
  }
  path.resize(copied);

  const int rc = idx->write(path);
  if (rc != 0) {
    throw_status(env, rc);
    return nullptr;
  }
  js_value_t* u = nullptr;
  js_get_undefined(env, &u);
  return u;
}

// Generic int32 getter for len/dim/bitWidth.
template <int (VectorIndex::*Fn)() const noexcept>
js_value_t* idx_int_getter(js_env_t* env, js_callback_info_t* info) {
  size_t argc = 1;
  js_value_t* argv[1] = { nullptr };
  if (js_get_callback_info(env, info, &argc, argv, nullptr, nullptr) != 0) {
    return nullptr;
  }
  VectorIndex* idx = unwrap(env, argv[0]);
  if (idx == nullptr) { return nullptr; }
  js_value_t* result = nullptr;
  js_create_int32(env, (idx->*Fn)(), &result);
  return result;
}

} // namespace

namespace qvac_lib_inference_addon_embed::vector_index {

void registerBindings(js_env_t* env, js_value_t* exports) {
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define V(name, fn)                                                            \
  do {                                                                         \
    js_value_t* val = nullptr;                                                 \
    if (js_create_function(env, name, -1, fn, nullptr, &val) != 0) {           \
      return;                                                                  \
    }                                                                          \
    if (js_set_named_property(env, exports, name, val) != 0) {                 \
      return;                                                                  \
    }                                                                          \
  } while (0)

  V("idx_create",    idx_create);
  V("idx_load",      idx_load);
  V("idx_add",       idx_add);
  V("idx_search",    idx_search);
  V("idx_remove",    idx_remove);
  V("idx_contains",  idx_contains);
  V("idx_prepare",   idx_prepare);
  V("idx_write",     idx_write);
  V("idx_len",       (idx_int_getter<&VectorIndex::len>));
  V("idx_dim",       (idx_int_getter<&VectorIndex::dim>));
  V("idx_bit_width", (idx_int_getter<&VectorIndex::bitWidth>));
#undef V
// NOLINTEND(cppcoreguidelines-macro-usage)
}

} // namespace qvac_lib_inference_addon_embed::vector_index

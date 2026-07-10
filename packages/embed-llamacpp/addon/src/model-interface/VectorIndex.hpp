#pragma once
//
// RAII C++ wrapper around fabric's `ggml_vec_index_t*` C handle. Provides
// typed methods + status-aware accessors. Lifecycle isolated from
// LlamaLazyInitializeBackend / BertModel by construction: this header
// only depends on the ggml-base vector-index C API.

#include <ggml-vector-index.h>

#include <cstdint>
#include <string>

namespace qvac_lib_infer_llamacpp_embed {

class VectorIndex {
public:
  // Construct a fresh empty index. Throws std::invalid_argument on bad
  // dims / bit_width.
  VectorIndex(int dim, int bitWidth);

  // Adopt an already-opened native handle (used by static load).
  explicit VectorIndex(ggml_vec_index_t* handle) noexcept;

  ~VectorIndex();

  VectorIndex(const VectorIndex&) = delete;
  VectorIndex& operator=(const VectorIndex&) = delete;
  VectorIndex(VectorIndex&& other) noexcept;
  VectorIndex& operator=(VectorIndex&& other) noexcept;

  // Returns 0 on success, ggml_vec_index_error on failure (e.g. duplicate).
  int add(const float* vectors, int n, const uint64_t* ids) noexcept;

  // Returns 1 / 0 (removed / not present), negative on error.
  int remove(uint64_t id) noexcept;

  [[nodiscard]] bool contains(uint64_t id) const noexcept;

  void prepare() noexcept;

  // Top-k search. Caller owns out arrays of size n_q * k.
  int search(
      const float* queries,
      int n_q,
      int k,
      float* outScores,
      uint64_t* outIds) const noexcept;

  // Persists to disk. Returns 0 on success.
  int write(const std::string& path) noexcept;

  // Reads from disk. On failure returns a wrapper whose `valid()` is false;
  // callers must check before using the instance.
  static VectorIndex load(const std::string& path) noexcept;

  // Stats.
  [[nodiscard]] int len() const noexcept;
  [[nodiscard]] int dim() const noexcept;
  [[nodiscard]] int bitWidth() const noexcept;

  // True if this instance owns a native handle (i.e. wasn't moved-from /
  // wasn't a failed load).
  [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }

  // Raw handle accessor for the JS binding's finalizer. Caller must not
  // free; ownership remains with this object.
  [[nodiscard]] ggml_vec_index_t* raw() const noexcept { return handle_; }

private:
  ggml_vec_index_t* handle_;
};

} // namespace qvac_lib_infer_llamacpp_embed

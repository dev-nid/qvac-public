#include "VectorIndex.hpp"

#include <stdexcept>
#include <utility>

namespace qvac_lib_infer_llamacpp_embed {

VectorIndex::VectorIndex(int dim, int bitWidth)
    : handle_(ggml_vec_index_create(dim, bitWidth)) {
  if (handle_ == nullptr) {
    throw std::invalid_argument(
        "ggml_vec_index_create rejected dim/bitWidth");
  }
}

VectorIndex::VectorIndex(ggml_vec_index_t* handle) noexcept : handle_(handle) {}

VectorIndex::~VectorIndex() {
  if (handle_ != nullptr) {
    ggml_vec_index_free(handle_);
    handle_ = nullptr;
  }
}

VectorIndex::VectorIndex(VectorIndex&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

VectorIndex& VectorIndex::operator=(VectorIndex&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) {
      ggml_vec_index_free(handle_);
    }
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

int VectorIndex::add(
    const float* vectors, int n, const uint64_t* ids) noexcept {
  return ggml_vec_index_add(handle_, vectors, n, ids);
}

int VectorIndex::remove(uint64_t id) noexcept {
  return ggml_vec_index_remove(handle_, id);
}

bool VectorIndex::contains(uint64_t id) const noexcept {
  return ggml_vec_index_contains(handle_, id) != 0;
}

void VectorIndex::prepare() noexcept { ggml_vec_index_prepare(handle_); }

int VectorIndex::search(
    const float* queries,
    int n_q,
    int k,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search(
      handle_, queries, n_q, k, outScores, outIds);
}

int VectorIndex::write(const std::string& path) noexcept {
  return ggml_vec_index_write(handle_, path.c_str());
}

VectorIndex VectorIndex::load(const std::string& path) noexcept {
  ggml_vec_index_t* raw = ggml_vec_index_load(path.c_str());
  return VectorIndex(raw);
}

int VectorIndex::len() const noexcept {
  return ggml_vec_index_len(handle_);
}

int VectorIndex::dim() const noexcept {
  return ggml_vec_index_dim(handle_);
}

int VectorIndex::bitWidth() const noexcept {
  return ggml_vec_index_bit_width(handle_);
}

} // namespace qvac_lib_infer_llamacpp_embed

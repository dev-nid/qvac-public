#include "VectorIndex.hpp"

#include <stdexcept>
#include <utility>

namespace qvac_lib_infer_llamacpp_embed {

VectorIndexFilter::VectorIndexFilter(
    ggml_vec_index_filter_t* handle) noexcept
    : handle_(handle) {}

VectorIndexFilter::~VectorIndexFilter() {
  if (handle_ != nullptr) {
    ggml_vec_index_filter_free(handle_);
    handle_ = nullptr;
  }
}

VectorIndexFilter::VectorIndexFilter(VectorIndexFilter&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = nullptr;
}

VectorIndexFilter& VectorIndexFilter::operator=(
    VectorIndexFilter&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) {
      ggml_vec_index_filter_free(handle_);
    }
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

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

int VectorIndex::addLogged(
    const float* vectors,
    int n,
    const uint64_t* ids,
    const std::string& deltaPath) noexcept {
  return ggml_vec_index_add_logged(
      handle_, vectors, n, ids, deltaPath.c_str());
}

int VectorIndex::remove(uint64_t id) noexcept {
  return ggml_vec_index_remove(handle_, id);
}

int VectorIndex::removeLogged(
    uint64_t id,
    const std::string& deltaPath) noexcept {
  return ggml_vec_index_remove_logged(handle_, id, deltaPath.c_str());
}

int VectorIndex::compact() noexcept {
  return ggml_vec_index_compact(handle_);
}

bool VectorIndex::contains(uint64_t id) const noexcept {
  return ggml_vec_index_contains(handle_, id) != 0;
}

void VectorIndex::prepare() noexcept { ggml_vec_index_prepare(handle_); }

int VectorIndex::buildIvf(int nLists, int nIter) noexcept {
  return ggml_vec_index_build_ivf(handle_, nLists, nIter);
}

int VectorIndex::prepareGpu() noexcept {
  return ggml_vec_index_prepare_gpu(handle_);
}

int VectorIndex::search(
    const float* queries,
    int n_q,
    int k,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search(
      handle_, queries, n_q, k, outScores, outIds);
}

int VectorIndex::searchFiltered(
    const float* queries,
    int n_q,
    int k,
    const uint64_t* allowedIds,
    int nAllowed,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_filtered(
      handle_, queries, n_q, k, allowedIds, nAllowed, outScores, outIds);
}

VectorIndexFilter VectorIndex::createFilter(
    const uint64_t* allowedIds,
    int nAllowed) const noexcept {
  ggml_vec_index_filter_t* raw = ggml_vec_index_filter_create(
      handle_, allowedIds, nAllowed);
  return VectorIndexFilter(raw);
}

int VectorIndex::searchPreparedFiltered(
    const VectorIndexFilter& filter,
    const float* queries,
    int n_q,
    int k,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_prepared_filtered(
      handle_, filter.raw(), queries, n_q, k, outScores, outIds);
}

int VectorIndex::searchGpu(
    const float* queries,
    int n_q,
    int k,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_gpu_topk(
      handle_, queries, n_q, k, outScores, outIds);
}

int VectorIndex::searchGpuPreparedFiltered(
    const VectorIndexFilter& filter,
    const float* queries,
    int n_q,
    int k,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_gpu_prepared_filtered_topk(
      handle_, filter.raw(), queries, n_q, k, outScores, outIds);
}

int VectorIndex::searchIvf(
    const float* queries,
    int n_q,
    int k,
    int nProbe,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_ivf(
      handle_, queries, n_q, k, nProbe, outScores, outIds);
}

int VectorIndex::searchGpuIvf(
    const float* queries,
    int n_q,
    int k,
    int nProbe,
    float* outScores,
    uint64_t* outIds) const noexcept {
  return ggml_vec_index_search_gpu_ivf_topk(
      handle_, queries, n_q, k, nProbe, outScores, outIds);
}

int VectorIndex::write(const std::string& path) noexcept {
  return ggml_vec_index_write(handle_, path.c_str());
}

int VectorIndex::compactDelta(
    const std::string& snapshotPath,
    const std::string& deltaPath) noexcept {
  return ggml_vec_index_compact_delta(
      handle_, snapshotPath.c_str(), deltaPath.c_str());
}

VectorIndex VectorIndex::load(const std::string& path) noexcept {
  ggml_vec_index_t* raw = ggml_vec_index_load(path.c_str());
  return VectorIndex(raw);
}

VectorIndex VectorIndex::loadWithDelta(
    const std::string& snapshotPath,
    const std::string& deltaPath) noexcept {
  ggml_vec_index_t* raw = ggml_vec_index_load_with_delta(
      snapshotPath.c_str(), deltaPath.c_str());
  return VectorIndex(raw);
}

VectorIndex VectorIndex::loadMmap(const std::string& path) noexcept {
  ggml_vec_index_t* raw = ggml_vec_index_load_mmap(path.c_str());
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

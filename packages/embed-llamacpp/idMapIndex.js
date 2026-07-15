'use strict'

// IdMapIndex: thin JS wrapper around the IdMapIndex N-API bindings exposed
// by the embed-llamacpp addon (vector-index-binding.cpp). The native side
// supports full f32 storage (`bitWidth: 32`) and production q8/q4 storage
// (`bitWidth: 8` or `4`) with CPU search against the selected representation.
//
// Lifecycle isolation: constructing IdMapIndex does NOT load any BERT
// model. It only resolves the native binding (loaded once per process)
// and calls idx_create, which is a tiny C-API call into ggml-base. Pair
// this with the assertion in the smoke test (no GGMLBert constructor
// runs while exercising the index).

const binding = require('./binding')

const HANDLE = Symbol('IdMapIndex.handle')
const FILTER_HANDLE = Symbol('IdMapIndexFilter.handle')
const FILTER_OWNER = Symbol('IdMapIndexFilter.owner')

function ensureHandle(self) {
  if (self[HANDLE] === null || self[HANDLE] === undefined) {
    throw new Error('IdMapIndex has been disposed')
  }
  return self[HANDLE]
}

function ensureFilterHandle(self) {
  if (self[FILTER_HANDLE] === null || self[FILTER_HANDLE] === undefined) {
    throw new Error('IdMapIndexFilter has been disposed')
  }
  return self[FILTER_HANDLE]
}

class IdMapIndexFilter {
  search(queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('IdMapIndexFilter.search: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new TypeError('IdMapIndexFilter.search: k must be a positive integer')
    }
    const filterHandle = ensureFilterHandle(this)
    return binding.idx_search_prepared_filtered(
      ensureHandle(this[FILTER_OWNER]),
      filterHandle,
      queries,
      k
    )
  }

  searchGpu(queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('IdMapIndexFilter.searchGpu: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0 || k > 64) {
      throw new TypeError('IdMapIndexFilter.searchGpu: k must be an integer between 1 and 64')
    }
    const filterHandle = ensureFilterHandle(this)
    return binding.idx_search_gpu_prepared_filtered(
      ensureHandle(this[FILTER_OWNER]),
      filterHandle,
      queries,
      k
    )
  }

  dispose() {
    if (this[FILTER_HANDLE] !== null && this[FILTER_HANDLE] !== undefined) {
      binding.idx_filter_dispose(this[FILTER_HANDLE])
      this[FILTER_HANDLE] = null
      this[FILTER_OWNER] = null
    }
    return Promise.resolve()
  }
}

/**
 * In-memory TurboVec-style vector index. Indexes vectors of fixed
 * dimensionality and stable external uint64 ids; supports top-k dot-product
 * search. Callers that need cosine similarity should L2-normalize vectors
 * before both insertion and query.
 */
class IdMapIndex {
  /**
   * @param {object} opts
   * @param {number} opts.dim - vector dimensionality (must be > 0)
   * @param {4|8|32} [opts.bitWidth=8] - 4 = q4, 8 = q8, 32 = f32 storage
   */
  constructor({ dim, bitWidth = 8 } = {}) {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new TypeError('IdMapIndex: dim must be a positive integer')
    }
    if (bitWidth !== 4 && bitWidth !== 8 && bitWidth !== 32) {
      throw new TypeError('IdMapIndex: bitWidth must be 4, 8, or 32')
    }
    this[HANDLE] = binding.idx_create({ dim, bitWidth })
  }

  /**
   * Load a persisted index from a .tvim file. Returns a fresh instance.
   * @param {string} path
   * @returns {Promise<IdMapIndex>}
   */
  static load(path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IdMapIndex.load: path must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load(path)
    return Promise.resolve(instance)
  }

  /**
   * Load a persisted .tvim file with mmap-backed vector storage. The returned
   * index is read-only for mutating operations.
   * @param {string} path
   * @returns {Promise<IdMapIndex>}
   */
  static loadMmap(path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IdMapIndex.loadMmap: path must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load_mmap(path)
    return Promise.resolve(instance)
  }

  /**
   * Load a full snapshot and replay an append-only delta log.
   * @param {string} snapshotPath
   * @param {string} deltaPath
   * @returns {Promise<IdMapIndex>}
   */
  static loadWithDelta(snapshotPath, deltaPath) {
    if (typeof snapshotPath !== 'string' || snapshotPath.length === 0) {
      throw new TypeError('IdMapIndex.loadWithDelta: snapshotPath must be a non-empty string')
    }
    if (typeof deltaPath !== 'string' || deltaPath.length === 0) {
      throw new TypeError('IdMapIndex.loadWithDelta: deltaPath must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load_with_delta(snapshotPath, deltaPath)
    return Promise.resolve(instance)
  }

  /**
   * Insert `n` vectors with their external ids. Throws on duplicate id or
   * dim mismatch; partial mutation is impossible (atomic add).
   * @param {Float32Array} vectors - length = n * dim
   * @param {BigUint64Array} ids   - length = n
   */
  addWithIds(vectors, ids) {
    if (!(vectors instanceof Float32Array)) {
      throw new TypeError('addWithIds: vectors must be a Float32Array')
    }
    if (!(ids instanceof BigUint64Array)) {
      throw new TypeError('addWithIds: ids must be a BigUint64Array')
    }
    if (vectors.length !== ids.length * this.dim) {
      throw new RangeError(
        `addWithIds: vectors.length (${vectors.length}) must equal ids.length (${ids.length}) * dim (${this.dim})`
      )
    }
    if (ids.length === 0) return
    binding.idx_add(ensureHandle(this), vectors, ids)
  }

  /**
   * Insert vectors and append the mutation to an incremental delta log.
   * @param {Float32Array} vectors - length = n * dim
   * @param {BigUint64Array} ids   - length = n
   * @param {string} deltaPath
   */
  addLogged(vectors, ids, deltaPath) {
    if (!(vectors instanceof Float32Array)) {
      throw new TypeError('addLogged: vectors must be a Float32Array')
    }
    if (!(ids instanceof BigUint64Array)) {
      throw new TypeError('addLogged: ids must be a BigUint64Array')
    }
    if (typeof deltaPath !== 'string' || deltaPath.length === 0) {
      throw new TypeError('addLogged: deltaPath must be a non-empty string')
    }
    if (vectors.length !== ids.length * this.dim) {
      throw new RangeError(
        `addLogged: vectors.length (${vectors.length}) must equal ids.length (${ids.length}) * dim (${this.dim})`
      )
    }
    if (ids.length === 0) return
    binding.idx_add_logged(ensureHandle(this), vectors, ids, deltaPath)
  }

  /**
   * Top-k search across `m` queries packed contiguously in `queries`.
   * Returns `{ scores, ids, m, k }` where scores/ids are typed arrays of
   * length m*k (row-major).
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  search(queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('search: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new TypeError('search: k must be a positive integer')
    }
    return binding.idx_search(ensureHandle(this), queries, k)
  }

  /**
   * Exact Metal/GPU top-k search. `prepareGpu()` must have run after the latest
   * mutation. Unsupported platforms throw so callers can fall back to `search()`.
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k - 1..64
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  searchGpu(queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('searchGpu: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0 || k > 64) {
      throw new TypeError('searchGpu: k must be an integer between 1 and 64')
    }
    return binding.idx_search_gpu(ensureHandle(this), queries, k)
  }

  /**
   * Top-k search restricted to `allowedIds`. Missing ids are ignored; an empty
   * allowlist returns only sentinel padding.
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k
   * @param {BigUint64Array} allowedIds
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  searchFiltered(queries, k, allowedIds) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('searchFiltered: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new TypeError('searchFiltered: k must be a positive integer')
    }
    if (!(allowedIds instanceof BigUint64Array)) {
      throw new TypeError('searchFiltered: allowedIds must be a BigUint64Array')
    }
    return binding.idx_search_filtered(ensureHandle(this), queries, k, allowedIds)
  }

  /**
   * Prepare an allowlist for repeated filtered searches. Any successful
   * mutation on this index invalidates existing prepared filters.
   * @param {BigUint64Array} allowedIds
   * @returns {IdMapIndexFilter}
   */
  prepareFilter(allowedIds) {
    if (!(allowedIds instanceof BigUint64Array)) {
      throw new TypeError('prepareFilter: allowedIds must be a BigUint64Array')
    }
    const filter = Object.create(IdMapIndexFilter.prototype)
    filter[FILTER_OWNER] = this
    filter[FILTER_HANDLE] = binding.idx_filter_create(ensureHandle(this), allowedIds)
    return filter
  }

  /**
   * Build IVF-flat approximate search state. Mutations invalidate this state.
   * @param {number} nLists
   * @param {number} [nIter=0]
   */
  buildIvf(nLists, nIter = 0) {
    if (!Number.isInteger(nLists) || nLists <= 0) {
      throw new TypeError('buildIvf: nLists must be a positive integer')
    }
    if (!Number.isInteger(nIter) || nIter < 0) {
      throw new TypeError('buildIvf: nIter must be a non-negative integer')
    }
    binding.idx_build_ivf(ensureHandle(this), nLists, nIter)
  }

  /**
   * IVF-flat ANN top-k search. `buildIvf()` must have run after the latest
   * mutation.
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k
   * @param {number} nProbe
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  searchIvf(queries, k, nProbe) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('searchIvf: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new TypeError('searchIvf: k must be a positive integer')
    }
    if (!Number.isInteger(nProbe) || nProbe <= 0) {
      throw new TypeError('searchIvf: nProbe must be a positive integer')
    }
    return binding.idx_search_ivf(ensureHandle(this), queries, k, nProbe)
  }

  /**
   * IVF-flat ANN Metal/GPU top-k search. `prepareGpu()` and `buildIvf()` must
   * have run after the latest mutation.
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k - 1..64
   * @param {number} nProbe
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  searchIvfGpu(queries, k, nProbe) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('searchIvfGpu: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0 || k > 64) {
      throw new TypeError('searchIvfGpu: k must be an integer between 1 and 64')
    }
    if (!Number.isInteger(nProbe) || nProbe <= 0) {
      throw new TypeError('searchIvfGpu: nProbe must be a positive integer')
    }
    return binding.idx_search_gpu_ivf(ensureHandle(this), queries, k, nProbe)
  }

  /**
   * Remove an entry by external id.
   * @param {bigint} id
   * @returns {boolean} true if removed, false if not present
   */
  remove(id) {
    if (typeof id !== 'bigint') {
      throw new TypeError('remove: id must be a bigint')
    }
    return binding.idx_remove(ensureHandle(this), id)
  }

  /**
   * Remove an entry and append the mutation to an incremental delta log.
   * @param {bigint} id
   * @param {string} deltaPath
   * @returns {boolean}
   */
  removeLogged(id, deltaPath) {
    if (typeof id !== 'bigint') {
      throw new TypeError('removeLogged: id must be a bigint')
    }
    if (typeof deltaPath !== 'string' || deltaPath.length === 0) {
      throw new TypeError('removeLogged: deltaPath must be a non-empty string')
    }
    return binding.idx_remove_logged(ensureHandle(this), id, deltaPath)
  }

  /** Physically remove deleted slots from the in-memory index. */
  compact() {
    binding.idx_compact(ensureHandle(this))
  }

  /**
   * @param {bigint} id
   * @returns {boolean}
   */
  contains(id) {
    if (typeof id !== 'bigint') {
      throw new TypeError('contains: id must be a bigint')
    }
    return binding.idx_contains(ensureHandle(this), id)
  }

  /** Placeholder for future cache warming. */
  prepare() {
    binding.idx_prepare(ensureHandle(this))
  }

  /**
   * Prepare optional Metal/GPU search cache. Mutations invalidate this state.
   * Unsupported platforms throw so callers can fall back to CPU search.
   */
  prepareGpu() {
    binding.idx_prepare_gpu(ensureHandle(this))
  }

  /**
   * Persist the index to disk in the checksummed .tvim v2 format.
   * @param {string} path
   */
  write(path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('write: path must be a non-empty string')
    }
    binding.idx_write(ensureHandle(this), path)
  }

  /**
   * Write a compacted full snapshot and reset the matching delta log.
   * @param {string} snapshotPath
   * @param {string} deltaPath
   */
  compactDelta(snapshotPath, deltaPath) {
    if (typeof snapshotPath !== 'string' || snapshotPath.length === 0) {
      throw new TypeError('compactDelta: snapshotPath must be a non-empty string')
    }
    if (typeof deltaPath !== 'string' || deltaPath.length === 0) {
      throw new TypeError('compactDelta: deltaPath must be a non-empty string')
    }
    binding.idx_compact_delta(ensureHandle(this), snapshotPath, deltaPath)
  }

  get length() {
    return binding.idx_len(ensureHandle(this))
  }
  get dim() {
    return binding.idx_dim(ensureHandle(this))
  }
  get bitWidth() {
    return binding.idx_bit_width(ensureHandle(this))
  }

  /**
   * Free native resources and mark the instance as unusable. The external's
   * finalizer remains as a safety net if callers forget to dispose.
   */
  dispose() {
    if (this[HANDLE] !== null && this[HANDLE] !== undefined) {
      binding.idx_dispose(this[HANDLE])
      this[HANDLE] = null
    }
    return Promise.resolve()
  }
}

IdMapIndex.Filter = IdMapIndexFilter
IdMapIndex.IdMapIndexFilter = IdMapIndexFilter

module.exports = IdMapIndex

'use strict'

// IdMapIndex: thin JS wrapper around the IdMapIndex N-API bindings exposed
// by the embed-llamacpp addon (vector-index-binding.cpp). The native side
// supports full f32 storage (`storage: 'f32'`), q8/q4 storage, and TurboVec
// q2/q4 storage with CPU search against the selected representation.
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
const INT32_MAX = 0x7fffffff
const BIT_WIDTH_BY_STORAGE = {
  f32: 32,
  q8: 8,
  q4: 4,
  'turbovec-q4': 4,
  'turbovec-q2': 2
}
const STORAGE_BY_BIT_WIDTH = {
  2: 'turbovec-q2',
  4: 'q4',
  8: 'q8',
  32: 'f32'
}

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

function isPositiveInt32(value) {
  return Number.isInteger(value) && value > 0 && value <= INT32_MAX
}

function isNonNegativeInt32(value) {
  return Number.isInteger(value) && value >= 0 && value <= INT32_MAX
}

function normalizeStorageOptions(opts) {
  const hasBitWidth = Object.prototype.hasOwnProperty.call(opts, 'bitWidth')
  const hasStorage = Object.prototype.hasOwnProperty.call(opts, 'storage')
  let bitWidth = hasBitWidth ? opts.bitWidth : 8
  let storage = hasStorage ? opts.storage : undefined

  if (storage !== undefined) {
    if (
      typeof storage !== 'string' ||
      !Object.prototype.hasOwnProperty.call(BIT_WIDTH_BY_STORAGE, storage)
    ) {
      throw new TypeError(
        "IdMapIndex: storage must be 'f32', 'q8', 'q4', 'turbovec-q4', or 'turbovec-q2'"
      )
    }
    if (!hasBitWidth) {
      bitWidth = BIT_WIDTH_BY_STORAGE[storage]
    } else if (bitWidth !== BIT_WIDTH_BY_STORAGE[storage]) {
      throw new TypeError('IdMapIndex: bitWidth does not match storage')
    }
  } else {
    storage = STORAGE_BY_BIT_WIDTH[bitWidth]
  }

  if (storage === undefined) {
    throw new TypeError('IdMapIndex: bitWidth must be 2, 4, 8, or 32')
  }

  return { bitWidth, storage }
}

class IdMapIndexFilter {
  constructor() {
    throw new TypeError('IdMapIndexFilter instances must be created by IdMapIndex.prepareFilter()')
  }

  search(queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('IdMapIndexFilter.search: queries must be a Float32Array')
    }
    if (!isPositiveInt32(k)) {
      throw new TypeError('IdMapIndexFilter.search: k must be a positive int32')
    }
    const filterHandle = ensureFilterHandle(this)
    return binding.idx_search_prepared_filtered(
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
   * @param {2|4|8|32} [opts.bitWidth=8] - 2 = TurboVec q2, 4 = q4, 8 = q8, 32 = f32 storage
   * @param {'f32'|'q8'|'q4'|'turbovec-q4'|'turbovec-q2'} [opts.storage] - explicit storage mode
   */
  constructor(opts = {}) {
    const { dim } = opts
    if (!isPositiveInt32(dim)) {
      throw new TypeError('IdMapIndex: dim must be a positive int32')
    }
    const { bitWidth, storage } = normalizeStorageOptions(opts)
    this[HANDLE] = binding.idx_create({ dim, bitWidth, storage })
  }

  /**
   * Load a persisted index from a .tvim file. Returns a fresh instance.
   * This is synchronous and may block for large persisted indexes.
   * @param {string} path
   * @returns {IdMapIndex}
   */
  static load(path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IdMapIndex.load: path must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load(path)
    return instance
  }

  /**
   * Load a persisted .tvim file with mmap-backed vector storage. The returned
   * index is read-only for mutating operations.
   * This is synchronous and may block for large persisted indexes.
   * @param {string} path
   * @returns {IdMapIndex}
   */
  static loadMmap(path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IdMapIndex.loadMmap: path must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load_mmap(path)
    return instance
  }

  /**
   * Load a full snapshot and replay an append-only delta log.
   * This is synchronous and may block for large persisted indexes.
   * If another writer appends to the same delta log, reload with
   * `IdMapIndex.loadWithDelta()` before writing more logged mutations.
   * @param {string} snapshotPath
   * @param {string} deltaPath
   * @returns {IdMapIndex}
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
    return instance
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
   * Use a single writer instance per delta log; stale writers are rejected
   * and should reload with `IdMapIndex.loadWithDelta()` before appending.
   * Added vectors are stored in the delta log as f32 payloads regardless of
   * `bitWidth`; compact snapshots carry the selected storage savings.
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
   * Exact search is linear in index size; use filtered or IVF search to
   * reduce the candidate set.
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
    if (!isPositiveInt32(k)) {
      throw new TypeError('search: k must be a positive int32')
    }
    return binding.idx_search(ensureHandle(this), queries, k)
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
    if (!isPositiveInt32(k)) {
      throw new TypeError('searchFiltered: k must be a positive int32')
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
   * IVF state is in-memory only; call this after `load()`, `loadMmap()`,
   * or `loadWithDelta()` before using `searchIvf()`.
   * @param {number} nLists
   * @param {number} [nIter=0]
   */
  buildIvf(nLists, nIter = 0) {
    if (!isPositiveInt32(nLists)) {
      throw new TypeError('buildIvf: nLists must be a positive int32')
    }
    if (!isNonNegativeInt32(nIter)) {
      throw new TypeError('buildIvf: nIter must be a non-negative int32')
    }
    binding.idx_build_ivf(ensureHandle(this), nLists, nIter)
  }

  /**
   * IVF-flat ANN top-k search. `buildIvf()` must have run after the latest
   * mutation or load.
   * Higher `nProbe` improves recall at the cost of latency; tune
   * `nLists`, `nIter`, and `nProbe` against your dataset.
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k
   * @param {number} nProbe
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  searchIvf(queries, k, nProbe) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('searchIvf: queries must be a Float32Array')
    }
    if (!isPositiveInt32(k)) {
      throw new TypeError('searchIvf: k must be a positive int32')
    }
    if (!isPositiveInt32(nProbe)) {
      throw new TypeError('searchIvf: nProbe must be a positive int32')
    }
    return binding.idx_search_ivf(ensureHandle(this), queries, k, nProbe)
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
   * Use a single writer instance per delta log; stale writers are rejected
   * and should reload with `IdMapIndex.loadWithDelta()` before appending.
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
   * Coordinate compaction with the same single writer that owns the delta log.
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
  }
}

IdMapIndex.IdMapIndex = IdMapIndex
IdMapIndex.Filter = IdMapIndexFilter
IdMapIndex.IdMapIndexFilter = IdMapIndexFilter

module.exports = IdMapIndex

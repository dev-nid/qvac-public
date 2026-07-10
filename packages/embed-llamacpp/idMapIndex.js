'use strict'

// IdMapIndex: thin JS wrapper around the IdMapIndex N-API bindings exposed
// by the embed-llamacpp addon (vector-index-binding.cpp). The native side
// keeps the index in RAM as full f32 vectors for the POC; quantization,
// SIMD, and GPU kernels are future work behind the same JS shape.
//
// Lifecycle isolation: constructing IdMapIndex does NOT load any BERT
// model. It only resolves the native binding (loaded once per process)
// and calls idx_create, which is a tiny C-API call into ggml-base. Pair
// this with the assertion in the smoke test (no GGMLBert constructor
// runs while exercising the index).

const binding = require('./binding')

const HANDLE = Symbol('IdMapIndex.handle')

function ensureHandle (self) {
  if (self[HANDLE] == null) {
    throw new Error('IdMapIndex has been disposed')
  }
  return self[HANDLE]
}

/**
 * In-memory TurboQuant-style ANN vector index. Indexes vectors of fixed
 * dimensionality and stable external uint64 ids; supports top-k cosine /
 * dot-product search.
 *
 * Scope: POC. Internals store full f32 vectors and do scalar dot products;
 * quantization, SIMD, and GPU paths come later behind the same JS API.
 */
class IdMapIndex {
  /**
   * @param {object} opts
   * @param {number} opts.dim - vector dimensionality (must be > 0)
   * @param {number} [opts.bitWidth=4] - reserved for future quantization
   */
  constructor ({ dim, bitWidth = 4 } = {}) {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new TypeError('IdMapIndex: dim must be a positive integer')
    }
    if (!Number.isInteger(bitWidth) || bitWidth <= 0 || bitWidth > 32) {
      throw new TypeError('IdMapIndex: bitWidth must be an integer in [1, 32]')
    }
    this[HANDLE] = binding.idx_create({ dim, bitWidth })
  }

  /**
   * Load a persisted index from a .tvim file. Returns a fresh instance.
   * @param {string} path
   * @returns {Promise<IdMapIndex>}
   */
  static async load (path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('IdMapIndex.load: path must be a non-empty string')
    }
    const instance = Object.create(IdMapIndex.prototype)
    instance[HANDLE] = binding.idx_load(path)
    return instance
  }

  /**
   * Insert `n` vectors with their external ids. Throws on duplicate id or
   * dim mismatch; partial mutation is impossible (atomic add).
   * @param {Float32Array} vectors - length = n * dim
   * @param {BigUint64Array} ids   - length = n
   */
  addWithIds (vectors, ids) {
    if (!(vectors instanceof Float32Array)) {
      throw new TypeError('addWithIds: vectors must be a Float32Array')
    }
    if (!(ids instanceof BigUint64Array)) {
      throw new TypeError('addWithIds: ids must be a BigUint64Array')
    }
    if (ids.length === 0) return
    if (vectors.length !== ids.length * this.dim) {
      throw new RangeError(
        `addWithIds: vectors.length (${vectors.length}) must equal ids.length (${ids.length}) * dim (${this.dim})`
      )
    }
    binding.idx_add(ensureHandle(this), vectors, ids)
  }

  /**
   * Top-k search across `m` queries packed contiguously in `queries`.
   * Returns `{ scores, ids, m, k }` where scores/ids are typed arrays of
   * length m*k (row-major).
   * @param {Float32Array} queries - length = m * dim
   * @param {number} k
   * @returns {{ scores: Float32Array, ids: BigUint64Array, m: number, k: number }}
   */
  search (queries, k) {
    if (!(queries instanceof Float32Array)) {
      throw new TypeError('search: queries must be a Float32Array')
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new TypeError('search: k must be a positive integer')
    }
    return binding.idx_search(ensureHandle(this), queries, k)
  }

  /**
   * Remove an entry by external id.
   * @param {bigint} id
   * @returns {boolean} true if removed, false if not present
   */
  remove (id) {
    if (typeof id !== 'bigint') {
      throw new TypeError('remove: id must be a bigint')
    }
    return binding.idx_remove(ensureHandle(this), id)
  }

  /**
   * @param {bigint} id
   * @returns {boolean}
   */
  contains (id) {
    if (typeof id !== 'bigint') {
      throw new TypeError('contains: id must be a bigint')
    }
    return binding.idx_contains(ensureHandle(this), id)
  }

  /** No-op for the POC. Placeholder for future cache warming. */
  prepare () {
    binding.idx_prepare(ensureHandle(this))
  }

  /**
   * Persist the index to disk in the .tvim v1 format.
   * @param {string} path
   */
  write (path) {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('write: path must be a non-empty string')
    }
    binding.idx_write(ensureHandle(this), path)
  }

  get length () { return binding.idx_len(ensureHandle(this)) }
  get dim () { return binding.idx_dim(ensureHandle(this)) }
  get bitWidth () { return binding.idx_bit_width(ensureHandle(this)) }

  /**
   * Mark the instance as unusable. The native handle is owned by a
   * JS-side external whose finalizer runs at GC time; this method only
   * drops the local reference so subsequent calls throw (`ensureHandle`).
   * No memory is reclaimed synchronously — that requires a finalizer
   * pass. Kept as a method for API stability; future versions may add
   * a safe early-free path (wrapper object + sentinel) without changing
   * the call sites.
   */
  async dispose () {
    this[HANDLE] = null
  }
}

module.exports = IdMapIndex

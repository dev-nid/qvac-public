export type IdMapIndexStorage = 'f32' | 'q8' | 'q4' | 'turbovec-q4' | 'turbovec-q2'

export interface IdMapIndexOptions {
  /**
   * Vector dimensionality (must be > 0). TurboVec storage additionally
   * requires a 64-bit target and a dimension divisible by 8 and <= 65,536.
   */
  dim: number
  /** Storage precision: 2 = TurboVec q2, 4 = q4, 8 = q8, 32 = full f32 storage. Defaults to 8. */
  bitWidth?: 2 | 4 | 8 | 32
  /** Explicit storage mode. Use `turbovec-q4` to distinguish TurboVec q4 from generic q4. */
  storage?: IdMapIndexStorage
}

export interface IdMapIndexSearchResult {
  /** Row-major dot-product scores: m * k. Higher = closer. */
  scores: Float32Array
  /** Row-major external ids: m * k. `UINT64_MAX` padding when index is shorter than k. */
  ids: BigUint64Array
  /** Number of query rows. */
  m: number
  /** Effective k (mirrors the requested k). */
  k: number
}

export class IdMapIndexFilter {
  private constructor()

  /** Search with the prepared allowlist. */
  search(queries: Float32Array, k: number): IdMapIndexSearchResult

  dispose(): void
}

export default class IdMapIndex {
  static Filter: typeof IdMapIndexFilter
  static IdMapIndex: typeof IdMapIndex
  static IdMapIndexFilter: typeof IdMapIndexFilter

  constructor(opts: IdMapIndexOptions)

  /** Open a persisted .tvim file written by `write()`. Synchronous; may block for large indexes. */
  static load(path: string): IdMapIndex

  /** Open a persisted .tvim file with mmap-backed vector storage. Synchronous; mutations fail. */
  static loadMmap(path: string): IdMapIndex

  /**
   * Open a persisted .tvim snapshot and replay an append-only .tvid delta log.
   * Synchronous; may block for large indexes. If another writer appends to the
   * same delta log, reload before writing more logged mutations.
   */
  static loadWithDelta(snapshotPath: string, deltaPath: string): IdMapIndex

  /**
   * Insert `n` vectors with stable external ids. Throws on duplicate id or
   * dim mismatch; mutation is atomic per call. `UINT64_MAX` is reserved for
   * search result padding and cannot be inserted.
   */
  addWithIds(vectors: Float32Array, ids: BigUint64Array): void

  /**
   * Insert vectors and append the mutation to an incremental .tvid delta log.
   * `UINT64_MAX` is reserved for search result padding and cannot be inserted.
   * Use a single writer instance per delta log; stale writers are rejected and
   * should reload with `IdMapIndex.loadWithDelta()` before appending.
   * Once bound to a delta log, use logged mutations and `compactDelta()` for
   * content changes. q4/q8 logs store native quantized payloads; TurboVec
   * indexes do not support logged mutations.
   */
  addLogged(vectors: Float32Array, ids: BigUint64Array, deltaPath: string): void

  /** Top-k search across `queries.length / dim` rows. Exact search is linear in index size. */
  search(queries: Float32Array, k: number): IdMapIndexSearchResult

  /** Top-k search restricted to the supplied allowed ids. */
  searchFiltered(
    queries: Float32Array,
    k: number,
    allowedIds: BigUint64Array
  ): IdMapIndexSearchResult

  /** Prepare an allowlist for repeated filtered searches. */
  prepareFilter(allowedIds: BigUint64Array): IdMapIndexFilter

  /**
   * Build IVF-flat approximate search state. Mutations invalidate this state.
   * IVF state is in-memory only; call this after `load()`, `loadMmap()`, or
   * `loadWithDelta()` before using `searchIvf()`.
   */
  buildIvf(nLists: number, nIter?: number): void

  /**
   * IVF-flat ANN top-k search. `buildIvf()` must have run after the latest mutation or load.
   * Higher `nProbe` improves recall at the cost of latency; tune `nLists`, `nIter`, and `nProbe` against your dataset.
   */
  searchIvf(queries: Float32Array, k: number, nProbe: number): IdMapIndexSearchResult

  /** Returns true if removed, false if not present. */
  remove(id: bigint): boolean

  /**
   * Remove an entry and append the mutation to an incremental .tvid delta log.
   * Use a single writer instance per delta log; stale writers are rejected and
   * should reload with `IdMapIndex.loadWithDelta()` before appending.
   */
  removeLogged(id: bigint, deltaPath: string): boolean

  /** Physically remove deleted slots from the in-memory index. */
  compact(): void

  contains(id: bigint): boolean

  /** Warm storage-specific caches; TurboVec prepares rotation/codebook state. */
  prepare(): void

  /** Persist to .tvim v2 (f32/q4/q8) or v3 (TurboVec); legacy v1 remains readable. */
  write(path: string): void

  /**
   * Write a compacted full snapshot and reset the matching .tvid delta log.
   * Coordinate compaction with the same single writer that owns the delta log.
   */
  compactDelta(snapshotPath: string, deltaPath: string): void

  readonly length: number
  readonly dim: number
  readonly bitWidth: 2 | 4 | 8 | 32

  dispose(): void
}

export { IdMapIndex }

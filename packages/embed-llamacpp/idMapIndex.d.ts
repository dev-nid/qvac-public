export interface IdMapIndexOptions {
  /** Vector dimensionality (must be > 0). */
  dim: number
  /** Storage precision: 4 = q4, 8 = q8, 32 = full f32 storage. Defaults to 8. */
  bitWidth?: 4 | 8 | 32
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

  /** Open a persisted .tvim snapshot and replay an append-only .tvid delta log. Synchronous; may block for large indexes. */
  static loadWithDelta(
    snapshotPath: string,
    deltaPath: string
  ): IdMapIndex

  /**
   * Insert `n` vectors with stable external ids. Throws on duplicate id or
   * dim mismatch; mutation is atomic per call. `UINT64_MAX` is reserved for
   * search result padding and cannot be inserted.
   */
  addWithIds(vectors: Float32Array, ids: BigUint64Array): void

  /**
   * Insert vectors and append the mutation to an incremental .tvid delta log.
   * `UINT64_MAX` is reserved for search result padding and cannot be inserted.
   */
  addLogged(
    vectors: Float32Array,
    ids: BigUint64Array,
    deltaPath: string
  ): void

  /** Top-k search across `queries.length / dim` rows. */
  search(queries: Float32Array, k: number): IdMapIndexSearchResult

  /** Top-k search restricted to the supplied allowed ids. */
  searchFiltered(
    queries: Float32Array,
    k: number,
    allowedIds: BigUint64Array
  ): IdMapIndexSearchResult

  /** Prepare an allowlist for repeated filtered searches. */
  prepareFilter(allowedIds: BigUint64Array): IdMapIndexFilter

  /** Build IVF-flat approximate search state. Mutations invalidate this state. */
  buildIvf(nLists: number, nIter?: number): void

  /** IVF-flat ANN top-k search. `buildIvf()` must have run after the latest mutation. */
  searchIvf(queries: Float32Array, k: number, nProbe: number): IdMapIndexSearchResult

  /** Returns true if removed, false if not present. */
  remove(id: bigint): boolean

  /** Remove an entry and append the mutation to an incremental .tvid delta log. */
  removeLogged(id: bigint, deltaPath: string): boolean

  /** Physically remove deleted slots from the in-memory index. */
  compact(): void

  contains(id: bigint): boolean

  /** Placeholder for cache warming / codebook resolution after bulk add. */
  prepare(): void

  /** Persist to disk (.tvim v2; legacy v1 files are still readable). */
  write(path: string): void

  /** Write a compacted full snapshot and reset the matching .tvid delta log. */
  compactDelta(snapshotPath: string, deltaPath: string): void

  readonly length: number
  readonly dim: number
  readonly bitWidth: 4 | 8 | 32

  dispose(): void
}

export { IdMapIndex }

import type { QvacResponse } from '@qvac/infer-base'
import type QvacLogger from '@qvac/logging'

export { QvacResponse }

export interface Addon {
  loadWeights(data: {
    filename: string
    chunk: Uint8Array | null
    completed: boolean
  }): Promise<void>
  activate(): Promise<void>
  runJob(input: { type: 'text' | 'sequences'; input?: string | string[] }): Promise<boolean>
  cancel(): Promise<void>
  unload(): Promise<void>
}

export type NumericLike = `${number}`

export interface GGMLConfig {
  device: 'gpu' | 'cpu'
  gpu_layers?: NumericLike
  batch_size?: NumericLike
  ctx_size?: NumericLike
  pooling?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
  attention?: 'causal' | 'non-causal'
  embd_normalize?: NumericLike
  flash_attn?: 'on' | 'off' | 'auto'
  'main-gpu'?: NumericLike | 'integrated' | 'dedicated'
  'split-mode'?: 'none' | 'layer' | 'row'
  'tensor-split'?: string
  verbosity?: NumericLike
  /** Writable directory for OpenCL kernel binary cache. Required on Android for fast GPU startup. */
  openclCacheDir?: string
  [key: string]: string | number | boolean | string[] | undefined
}

export interface GGMLBertArgs {
  files: { model: string[] }
  config?: GGMLConfig
  logger?: QvacLogger | Console | null
  opts?: { stats?: boolean }
}

export interface AddonConfigurationParams {
  path: string
  config: GGMLConfig
  backendsDir?: string
}

export interface RuntimeStats {
  total_tokens: number
  total_time_ms: number
  tokens_per_second?: number
  batch_size: number
  trained_context_size: number
  context_size: number
  backendDevice: 'cpu' | 'gpu'
}

export default class GGMLBert {
  protected addon: Addon | null
  opts: { stats?: boolean }
  logger: QvacLogger
  state: { configLoaded: boolean }

  constructor(args: GGMLBertArgs)

  load(): Promise<void>
  run(text: string | string[]): Promise<QvacResponse>
  unload(): Promise<void>
  cancel(): Promise<void>
  getState(): { configLoaded: boolean }
}

export { GGMLBert }

export class BertInterface implements Addon {
  constructor(
    binding: unknown,
    configurationParams: AddonConfigurationParams,
    outputCb: (addon: unknown, event: string, data: unknown, error?: Error) => void
  )

  loadWeights(data: {
    filename: string
    chunk: Uint8Array | null
    completed: boolean
  }): Promise<void>
  activate(): Promise<void>
  runJob(input: { type: 'text' | 'sequences'; input?: string | string[] }): Promise<boolean>
  cancel(): Promise<void>
  unload(): Promise<void>
}

/** Returns the first shard (matching `-NNNNN-of-MMMMM.gguf`) or the sole entry for single-file models. */
export function pickPrimaryGgufPath(files: string[]): string

// ---------------------------------------------------------------------------
// IdMapIndex (turbovec)
// ---------------------------------------------------------------------------

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

  /** Metal/GPU search with the prepared allowlist. `prepareGpu()` must have run on the owner index. */
  searchGpu(queries: Float32Array, k: number): IdMapIndexSearchResult

  dispose(): Promise<void>
}

export class IdMapIndex {
  constructor(opts: IdMapIndexOptions)

  /** Open a persisted .tvim file written by `write()`. */
  static load(path: string): Promise<IdMapIndex>

  /** Open a persisted .tvim file with mmap-backed vector storage. Mutations fail. */
  static loadMmap(path: string): Promise<IdMapIndex>

  /** Open a persisted .tvim snapshot and replay an append-only .tvid delta log. */
  static loadWithDelta(snapshotPath: string, deltaPath: string): Promise<IdMapIndex>

  /**
   * Insert `n` vectors with stable external ids. Throws on duplicate id or
   * dim mismatch; mutation is atomic per call.
   */
  addWithIds(vectors: Float32Array, ids: BigUint64Array): void

  /** Insert vectors and append the mutation to an incremental .tvid delta log. */
  addLogged(vectors: Float32Array, ids: BigUint64Array, deltaPath: string): void

  /** Top-k search across `queries.length / dim` rows. */
  search(queries: Float32Array, k: number): IdMapIndexSearchResult

  /** Exact Metal/GPU top-k search. `prepareGpu()` must have run after the latest mutation. */
  searchGpu(queries: Float32Array, k: number): IdMapIndexSearchResult

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

  /** IVF-flat ANN Metal/GPU top-k search. `prepareGpu()` and `buildIvf()` must be current. */
  searchIvfGpu(queries: Float32Array, k: number, nProbe: number): IdMapIndexSearchResult

  /** Returns true if removed, false if not present. */
  remove(id: bigint): boolean

  /** Remove an entry and append the mutation to an incremental .tvid delta log. */
  removeLogged(id: bigint, deltaPath: string): boolean

  /** Physically remove deleted slots from the in-memory index. */
  compact(): void

  contains(id: bigint): boolean

  /** Placeholder for cache warming / codebook resolution after bulk add. */
  prepare(): void

  /** Prepare optional Metal/GPU search cache. Mutations invalidate this state. */
  prepareGpu(): void

  /** Persist to disk (.tvim v2; legacy v1 files are still readable). */
  write(path: string): void

  /** Write a compacted full snapshot and reset the matching .tvid delta log. */
  compactDelta(snapshotPath: string, deltaPath: string): void

  readonly length: number
  readonly dim: number
  readonly bitWidth: 4 | 8 | 32

  dispose(): Promise<void>
}

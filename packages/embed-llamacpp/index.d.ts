import type { QvacResponse } from '@qvac/infer-base'
import type QvacLogger from '@qvac/logging'

export { QvacResponse }

export interface Addon {
  loadWeights(data: { filename: string; chunk: Uint8Array | null; completed: boolean }): Promise<void>
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

  loadWeights(data: { filename: string; chunk: Uint8Array | null; completed: boolean }): Promise<void>
  activate(): Promise<void>
  runJob(input: { type: 'text' | 'sequences'; input?: string | string[] }): Promise<boolean>
  cancel(): Promise<void>
  unload(): Promise<void>
}

/** Returns the first shard (matching `-NNNNN-of-MMMMM.gguf`) or the sole entry for single-file models. */
export function pickPrimaryGgufPath(files: string[]): string

// ---------------------------------------------------------------------------
// IdMapIndex (turbovec POC)
// ---------------------------------------------------------------------------

export interface IdMapIndexOptions {
  /** Vector dimensionality (must be > 0). */
  dim: number
  /** Reserved for future quantization; POC stores full f32 internally. */
  bitWidth?: number
}

export interface IdMapIndexSearchResult {
  /** Row-major scores: m * k. Higher = closer. */
  scores: Float32Array
  /** Row-major external ids: m * k. `UINT64_MAX` padding when index is shorter than k. */
  ids: BigUint64Array
  /** Number of query rows. */
  m: number
  /** Effective k (mirrors the requested k). */
  k: number
}

export class IdMapIndex {
  constructor(opts: IdMapIndexOptions)

  /** Open a persisted .tvim file written by `write()`. */
  static load(path: string): Promise<IdMapIndex>

  /**
   * Insert `n` vectors with stable external ids. Throws on duplicate id or
   * dim mismatch; mutation is atomic per call.
   */
  addWithIds(vectors: Float32Array, ids: BigUint64Array): void

  /** Top-k search across `queries.length / dim` rows. */
  search(queries: Float32Array, k: number): IdMapIndexSearchResult

  /** Returns true if removed, false if not present. */
  remove(id: bigint): boolean
  contains(id: bigint): boolean

  /** No-op for the POC. */
  prepare(): void

  /** Persist to disk (.tvim v1). */
  write(path: string): void

  readonly length: number
  readonly dim: number
  readonly bitWidth: number

  dispose(): Promise<void>
}

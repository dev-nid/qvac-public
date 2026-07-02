import type { QvacResponse } from '@qvac/infer-base'
import type QvacLogger from '@qvac/logging'

export type NumericLike = number | `${number}`

export interface AddonMessage {
  type: 'text'
  input: string
  prefill?: boolean
  /**
   * Per-call sampling overrides forwarded by `LlmLlamacpp.run()` from
   * `RunOptions.generationParams`. Carried on the `text` message and consumed
   * by the native binding so each `runJob` can use a different temp / top_p /
   * seed / etc. without re-loading the model.
   */
  generationParams?: GenerationParams
  cacheKey?: string
  saveCacheToDisk?: boolean
}
export interface AddonMediaMessage {
  type: 'media'
  content: Uint8Array
}
export type AddonRunJobMessage = AddonMessage | AddonMediaMessage

export interface Addon {
  loadWeights(data: { filename: string; chunk: Uint8Array | null; completed: boolean }, logger?: QvacLogger): Promise<void>
  activate(): Promise<void>
  /** Single-request admission: resolves `true` if accepted, `false` if busy. */
  runJob(data: AddonRunJobMessage[]): Promise<boolean>
  /** Batch admission: resolves the accepted flag plus the assigned sequence ids. */
  runJob(data: AddonBatchRunItem[]): Promise<AddonBatchRunResult>
  cancel(): Promise<void>
  finetune?(params: FinetuneOptions): Promise<boolean>
  unload(): Promise<void>
}

export interface AddonBatchRunItem {
  /** Optional caller-supplied id; the native binding auto-assigns one when omitted. */
  id?: string
  messages: AddonRunJobMessage[]
}

export interface AddonBatchRunResult {
  accepted: boolean
  ids: string[]
}

export interface LlamaConfig {
  device?: string
  gpu_layers?: NumericLike
  ctx_size?: NumericLike
  system_prompt?: string
  lora?: string
  temp?: NumericLike
  top_p?: NumericLike
  top_k?: NumericLike
  predict?: NumericLike
  seed?: NumericLike
  no_mmap?: boolean | ''
  reverse_prompt?: string
  repeat_penalty?: NumericLike
  presence_penalty?: NumericLike
  frequency_penalty?: NumericLike
  tools?: boolean | string
  verbosity?: NumericLike
  n_discarded?: NumericLike
  'main-gpu'?: NumericLike | string
  /** How to split the model across GPUs: 'none' (default, single GPU), 'layer' (pipeline parallelism), 'row' (tensor parallelism). */
  'split-mode'?: 'none' | 'layer' | 'row'
  /** Proportions for distributing layers/rows across GPUs (e.g. '1,1' for equal split, '3,1' for 75/25). */
  'tensor-split'?: string
  'cache-type-k'?: string
  'cache-type-v'?: string
  /** Writable directory for OpenCL kernel binary cache. Required on Android for fast GPU startup. */
  openclCacheDir?: string
  /**
   * Reasoning channel budget. `-1` (default) leaves the model's reasoning
   * channel on; `0` disables it; any positive integer caps the reasoning
   * channel at that many tokens (the sampler force-emits `</think>` once
   * the budget is exhausted).
   */
  reasoning_budget?: number | `${number}`
  /**
   * Number of concurrent sequence slots for continuous-batching (`--parallel` /
   * `n_parallel` in llama.cpp). Values `>= 2` activate the continuous-batch
   * scheduler so the prompts of a single batch-array `run()` call are decoded
   * together across slots; separate top-level `run()` calls are not batched
   * (only one response is active at a time). Default `1` (sequential, batching
   * disabled).
   */
  parallel?: NumericLike
  [key: string]: string | number | boolean | string[] | undefined
}

export interface LlmLlamacppArgs {
  files: { model: string[]; projectionModel?: string }
  config: LlamaConfig
  logger?: QvacLogger | Console | null
  opts?: { stats?: boolean }
}

export interface UserTextMessage {
  role: 'system' | 'assistant' | 'user' | 'tool' | 'session' | string
  content: string
  type?: undefined
  [key: string]: any
}

export interface UserMediaMessage {
  role: 'user'
  type: 'media'
  /**
   * Either the raw bytes of an image/audio/video file (`Uint8Array`) or an
   * absolute path to a file on disk (`string`). Path-mode is handled by the
   * C++ layer via `loadMedia()`; byte-mode takes the `parseMedia` path.
   */
  content: Uint8Array | string
}

export interface ChatFunctionDefinition {
  type: 'function'
  name: string
  description?: string
  parameters?: Record<string, any>
}

export type Message =
  | UserTextMessage
  | UserMediaMessage
  | ChatFunctionDefinition

export interface GenerationParams {
  temp?: number
  top_p?: number
  top_k?: number
  predict?: number
  seed?: number
  frequency_penalty?: number
  presence_penalty?: number
  repeat_penalty?: number
  /**
   * GBNF grammar applied per request to constrain sampling. Equivalent to
   * the load-time `--grammar` config but scoped to a single `run()` call;
   * the sampler is re-initialized with this grammar for the request and
   * the prior grammar is restored afterwards.
   *
   * `undefined` or an empty string is treated as "no override" and falls
   * through to whatever grammar was set at load time (typically none).
   *
   * Mutually exclusive with `json_schema` — passing both throws.
   */
  grammar?: string
  /**
   * JSON Schema applied per request to constrain sampling to valid JSON
   * matching the schema. Equivalent to the load-time `--json-schema`
   * config but scoped to a single `run()` call; the schema is converted
   * to GBNF natively (via llama.cpp's `json_schema_to_grammar()`) and
   * applied identically to `grammar`.
   *
   * Accepts either a JSON Schema object literal or a pre-stringified
   * JSON Schema. Mutually exclusive with `grammar` — passing both throws.
   */
  json_schema?: string | Record<string, unknown>
  /**
   * Per-request reasoning channel budget. `-1` keeps the model's reasoning
   * channel on; `0` disables it for this request; any positive integer caps
   * the reasoning channel at that many tokens. Equivalent to the load-time
   * `reasoning_budget` config but scoped to a single `run()` call; the prior
   * value is restored afterwards.
   */
  reasoning_budget?: number
  /**
   * When the model emits a reasoning block during generation (e.g.
   * `<think>...</think>` for the Qwen3 family, `<|channel>thought ...
   * <channel|>` for Gemma 4), drop those tokens from the KV cache at
   * end-of-generation so subsequent turns do not accumulate reasoning
   * history.
   *
   * Defaults to `true`: for reasoning-capable models the safer
   * default is to drop hidden reasoning blocks so that later turns
   * are not steered by internal reasoning that the user never sees.
   * Set to `false` to preserve reasoning tokens in the KV / SSM
   * cache across turns (e.g. chain-of-thought agents that want the
   * next turn to attend to prior reasoning, interpretability
   * tooling, or cache-reuse patterns that depend on the
   * reasoning-inclusive state). Supported on both text and
   * multimodal contexts. No-op for models without a recognised
   * reasoning channel.
   *
   * Recurrent / hybrid-SSM models (Qwen3.5, Qwen3-Next, Jamba,
   * Granite-Hybrid, ...) are supported: the recurrent half of the
   * memory module is snapshotted at the open marker, restored at end-
   * of-generation, and the post-reasoning tail is replayed through the
   * decoder so both KV halves stay consistent.
   *
   * Hard-failure contract on recurrent / hybrid-SSM models: if the
   * restore or replay step fails, the request is failed with a
   * `StatusError` rather than returning a partial answer that would
   * silently carry the reasoning span into subsequent turns.
   * Before throwing, the affected sequence's KV / SSM memory is
   * cleared and the driver's positional accounting is reset to a
   * clean, empty state, so the next request on the same context
   * starts from an empty sequence rather than contaminated SSM
   * hidden state. On the continuous-batch path this failure is
   * routed through the scheduler's error-recovery leg, which
   * deliberately does NOT persist the failed slot's cache: when the
   * request was configured with `cacheKey` + `saveCacheToDisk`, the
   * last known-good on-disk cache is preserved rather than being
   * overwritten with the post-failure state.
   *
   * Pure-attention models never take the restore/replay path, so
   * this hard-fail case does not apply to them; a rejected
   * attention `seq_rm` leaves the KV untouched and the request
   * simply completes without the compaction.
   *
   * Soft failures (snapshot capture on a recurrent / hybrid-SSM
   * model) do NOT fail the request. There are two sub-cases and
   * they surface through the same `RuntimeStats.thinkingCompactionFailed`
   * counter:
   *
   *   * Open-marker snapshot failed: end-of-generation compaction
   *     is skipped for this turn. Live cache is untouched, so the
   *     turn's answer is delivered normally, but the reasoning
   *     span remains in the KV / SSM cache. If the caller has
   *     `saveCacheToDisk` on the request, that un-compacted state
   *     is persisted to disk — the "remove" side of
   *     `remove_thinking_from_context` did not run for this turn.
   *   * Batch admission rollback-anchor snapshot failed: only
   *     cancel rollback for that turn degrades to a no-op — a
   *     subsequent cancel cannot roll the recurrent half back to
   *     the pre-request cursor. If the turn is not cancelled,
   *     live memory and any subsequent save are unaffected.
   */
  remove_thinking_from_context?: boolean
}

export interface RunOptions {
  prefill?: boolean
  generationParams?: GenerationParams
  cacheKey?: string
  saveCacheToDisk?: boolean
}

export interface BatchPrompt {
  id?: string
  prompt: Message[]
  runOptions?: RunOptions
}

export interface BatchOutputChunk {
  id: string
  chunk: string
}

export interface BatchResult {
  id: string
  output: string
}

export interface BatchResponse extends QvacResponse {
  ids: string[]
  on(event: 'output', cb: (chunk: BatchOutputChunk) => void): this
  onUpdate(cb: (chunk: BatchOutputChunk) => void): this
  await(): Promise<BatchResult[]>
}

export interface RuntimeStats {
  TTFT: number
  TPS: number
  ppTPS: number
  /** Final cache tokens for single requests, or the sum across completed batch slots. */
  CacheTokens: number
  generatedTokens: number
  promptTokens: number
  /** Context-window slides for single requests, or the sum across completed batch slots. */
  contextSlides: number
  /**
   * Number of `<think>` (or model-equivalent) reasoning blocks dropped
   * from the KV cache at end-of-generation by the
   * `remove_thinking_from_context` feature. Per-inference for single
   * requests; summed across completed slots for batch requests. 0 when
   * the model has no recognised reasoning channel, when the feature
   * was disabled per-request, or when no reasoning blocks were emitted.
   */
  thinkingBlockDiscards: number
  /**
   * Number of thinking-block compaction failures on recurrent /
   * hybrid-SSM models in the most recent generation. Always 0 for
   * pure-attention models (where the snapshot + replay step never
   * runs). Per-inference for single requests; summed across slots
   * for batch requests.
   *
   * Covers two distinct failure classes; check the request outcome
   * to disambiguate:
   *
   *   1. Hard failure: the end-of-generation restore or replay step
   *      failed. The request itself was failed with a `StatusError`
   *      (see `remove_thinking_from_context` for the reset/preserve
   *      contract). If the caller catches that error and reads
   *      `runtimeStats`, this counter reflects the failed turn.
   *   2. Soft failure — the request still completes and delivers an
   *      answer. Two sub-cases with different downstream effects:
   *      * Open-marker snapshot failed. End-of-generation
   *        thinking-block compaction is skipped for this turn, so
   *        the reasoning span remains in the live KV / SSM cache
   *        and, if the request has `saveCacheToDisk`, it is
   *        persisted to disk. Callers that rely on
   *        `remove_thinking_from_context` should treat a bump
   *        here as "the remove did not run this turn".
   *      * Batch admission rollback-anchor snapshot failed. Only
   *        cancel rollback for that turn degrades to a no-op; if
   *        the turn is not cancelled, live memory and any
   *        subsequent save are unaffected.
   */
  thinkingCompactionFailed: number
  /**
   * Average active sequences decoded together during the last request,
   * including overlapping requests from other callers.
   */
  avgConcurrentSeq: number
  backendDevice: 'cpu' | 'gpu'
}

export interface FinetuneValidationNone {
  type: 'none'
}

export interface FinetuneValidationSplit {
  type: 'split'
  /** Fraction of training data to hold out for validation (0–1). Default 0.05. */
  fraction?: number
}

export interface FinetuneValidationDataset {
  type: 'dataset'
  /** Path to a separate eval dataset file. Must differ from trainDatasetDir. */
  path: string
}

export type FinetuneValidation =
  | FinetuneValidationNone
  | FinetuneValidationSplit
  | FinetuneValidationDataset

export interface FinetuneOptions {
  /** Path to training dataset file (.jsonl for SFT, .txt for causal). */
  trainDatasetDir: string
  /** How to run validation. */
  validation: FinetuneValidation
  /** Directory (or file path ending in .gguf) for the final LoRA adapter. */
  outputParametersDir: string
  /** Number of training epochs. Default 1. */
  numberOfEpochs?: number
  /** Initial learning rate. Default 1e-4. */
  learningRate?: number
  /** Training sequence length. Default 128. */
  contextLength?: number
  /** Backend n_batch (tokens per batch). Must be >= microBatchSize and divisible by it. Default 128. */
  batchSize?: number
  /** Backend n_ubatch (micro-batch size). Must be <= batchSize. Default 128. */
  microBatchSize?: number
  /** Use SFT (chat) mode when true; causal (next-token) when false. Default false. */
  assistantLossOnly?: boolean
  /** Comma-separated LoRA target modules (e.g. 'attn_q,attn_k,attn_v,attn_o'). Default: attention Q/K/V/O. */
  loraModules?: string
  /** LoRA rank. Default 8. */
  loraRank?: number
  /** LoRA alpha (scaling factor). Default 16.0. */
  loraAlpha?: number
  /** LoRA init standard deviation. Default 0.02. */
  loraInitStd?: number
  /** Seed for LoRA weight initialization (0 = non-deterministic). Default 42. */
  loraSeed?: number
  /** Directory for checkpoints. Default './checkpoints'. */
  checkpointSaveDir?: string
  /** Save a checkpoint every N optimizer steps (0 = only on pause). Default 0. */
  checkpointSaveSteps?: number
  /** Path to a custom chat template file (for SFT). */
  chatTemplatePath?: string
  /** Learning rate scheduler: 'constant', 'cosine', or 'linear'. Default 'cosine'. */
  lrScheduler?: 'constant' | 'cosine' | 'linear'
  /** Minimum learning rate (for cosine/linear schedulers). Default 0. */
  lrMin?: number
  /** Warmup ratio (0–1). Requires warmupRatioSet: true. Default 0.1. */
  warmupRatio?: number
  /** When true, compute warmup steps from warmupRatio. */
  warmupRatioSet?: boolean
  /** Explicit warmup steps (used when warmupStepsSet is true). Default 0. */
  warmupSteps?: number
  /** When true, use warmupSteps directly instead of ratio. */
  warmupStepsSet?: boolean
  /** Weight decay. Default 0.01. */
  weightDecay?: number
}

export interface FinetuneProgressStats {
  is_train: boolean
  loss: number
  loss_uncertainty: number
  accuracy: number
  accuracy_uncertainty: number
  global_steps: number
  current_epoch: number
  current_batch: number
  total_batches: number
  elapsed_ms: number
  eta_ms: number
}

export interface FinetuneHandle {
  on(event: 'stats', cb: (stats: FinetuneProgressStats) => void): this
  removeListener(event: 'stats', cb: (stats: FinetuneProgressStats) => void): this
  await(): Promise<FinetuneResult>
}

export interface FinetuneStats {
  train_loss?: number
  train_loss_uncertainty?: number
  val_loss?: number
  val_loss_uncertainty?: number
  train_accuracy?: number
  train_accuracy_uncertainty?: number
  val_accuracy?: number
  val_accuracy_uncertainty?: number
  learning_rate?: number
  global_steps: number
  epochs_completed: number
}

export interface FinetuneResult {
  op: 'finetune'
  status: 'COMPLETED' | 'PAUSED'
  stats?: FinetuneStats
}

export default class LlmLlamacpp {
  protected addon: Addon | null
  opts: { stats?: boolean }
  logger: QvacLogger
  state: { configLoaded: boolean }

  constructor(args: LlmLlamacppArgs)

  load(): Promise<void>
  run(prompt: Message[], runOptions?: RunOptions): Promise<QvacResponse>
  run(prompt: (Message[] | BatchPrompt)[]): Promise<BatchResponse>
  finetune(finetuningOptions: FinetuneOptions): Promise<FinetuneHandle>
  cancel(): Promise<void>
  pause(): Promise<void>
  unload(): Promise<void>
  getState(): { configLoaded: boolean }
}

export { QvacResponse, FinetuneHandle, FinetuneProgressStats, FinetuneOptions, FinetuneValidation }

/** Returns the first shard (matching `-NNNNN-of-MMMMM.gguf`) or the sole entry for single-file models. */
export function pickPrimaryGgufPath(files: string[]): string

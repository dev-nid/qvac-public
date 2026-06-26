'use strict'

const path = require('bare-path')
const { ensureModel, safeTest } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const os = require('bare-os')
const LlmLlamacpp = require('../../index.js')

const isDarwinX64 = os.platform() === 'darwin' && os.arch() === 'x64'
const isLinuxArm64 = os.platform() === 'linux' && os.arch() === 'arm64'
const isWindowsX64 = os.platform() === 'win32' && os.arch() === 'x64'
const useCpu = isLinuxArm64

const MODEL = {
  name: 'Qwen3-0.6B-Q8_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf'
}

// Qwen3.5 is a separate family checkpoint: the PR widened reasoning detection
// from exact-match `qwen3` to a `qwen3*` prefix to cover it, and 3.5 is known
// to drive the KV cache differently (iM-RoPE / longer thinking traces), so the
// compaction path needs its own end-to-end coverage and not just the
// architecture-string unit test.
const QWEN35_MODEL = {
  name: 'Qwen3.5-0.8B-Q8_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf'
}

async function setupReasoningModel (t, toolsEnabled, opts = {}) {
  const { modelDef = MODEL, configOverrides = {} } = opts
  const [modelName, dirPath] = await ensureModel({
    modelName: modelDef.name,
    downloadUrl: modelDef.url
  })

  const modelPath = path.join(dirPath, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const config = {
    ctx_size: '4096',
    n_predict: '1024',
    seed: '50',
    gpu_layers: '999',
    temp: '0',
    top_p: '1',
    device: useCpu ? 'cpu' : 'gpu',
    verbosity: '2',
    tools: toolsEnabled ? 'true' : 'false',
    ...configOverrides
  }

  const inference = new LlmLlamacpp({
    files: { model: [modelPath] },
    config,
    logger: console,
    opts: { stats: true }
  })

  await inference.load()

  t.teardown(async () => {
    try {
      specLogger.release()
      if (inference) await inference.unload()
    } catch (err) {
      // Ignore cleanup errors
    }
  })

  return { inference }
}

// Shared helper: Run a completion and collect response
async function runCompletion (inference, messages, runOptions) {
  const result = await inference.run(messages, runOptions)
  let response = ''
  await result
    .onUpdate(token => {
      response += token
    })
    .await()
  return response
}

// Shared helper: Run a completion and return both response text + runtime stats.
async function runCompletionWithStats (inference, messages, runOptions) {
  const result = await inference.run(messages, runOptions)
  let response = ''
  await result
    .onUpdate(token => { response += token })
    .await()
  return { response, stats: result.stats || {} }
}

const toNumber = value => typeof value === 'number' ? value : Number(value || 0)

// Shared helper: Verify reasoning tags in response
function verifyReasoningTags (t, response, testName) {
  // Qwen3 models use <think> tags in output
  const hasOpeningTag = response.includes('<think>')
  const hasClosingTag = response.includes('</think>')
  t.ok(hasOpeningTag,
    `${testName} should contain opening reasoning tag`)
  t.ok(hasClosingTag,
    `${testName} should contain closing reasoning tag`)
  t.ok(response.length > 100,
    `${testName} should generate substantial output`)
}

// Shared helper: Verify generation continued after reasoning
function verifyContinuedAfterReasoning (t, response, testName) {
  const thinkCloseIndex = response.indexOf('</think>')
  if (thinkCloseIndex === -1) {
    t.fail(`No </think> tag found in ${testName}`)
    return false
  }

  const textAfterThink = response.substring(thinkCloseIndex + '</think>'.length).trim()
  t.ok(textAfterThink.length > 0,
    `Generation should continue after </think> tag (${testName})`)
  return textAfterThink.length > 0
}

// Shared helper: Create initial messages for reasoning test
function createInitialMessages () {
  return [
    {
      role: 'system',
      content: 'You are an AI assistant. Always provide a clear answer after thinking'
    },
    {
      role: 'user',
      content: 'what are you thinking'
    }
  ]
}

// Shared helper: Create follow-up messages
function createFollowUpMessages (initialMessages, previousResponse) {
  return [
    ...initialMessages,
    {
      role: 'assistant',
      content: previousResponse
    },
    {
      role: 'user',
      content: 'what is new'
    }
  ]
}

safeTest('reasoning tag EOS replacement works with tools=false', {
  skip: isDarwinX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  // First completion - should work correctly
  const messages1 = createInitialMessages()
  const response1 = await runCompletion(inference, messages1)
  t.comment(`First completion (tools=false, len=${response1.length}):\n${response1}`)
  verifyReasoningTags(t, response1, 'First completion')

  // Second completion - this is where the fix should activate
  const messages2 = createFollowUpMessages(messages1, response1)
  const response2 = await runCompletion(inference, messages2)
  t.comment(`Second completion (tools=false, len=${response2.length}):\n${response2}`)

  verifyReasoningTags(t, response2, 'Second completion')

  // Verify the fix worked: generation continued after reasoning
  verifyContinuedAfterReasoning(t, response2, 'tools=false')
})

safeTest('reasoning tag EOS replacement works with tools=true', {
  skip: isDarwinX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, true)

  // First completion - should work correctly
  const messages1 = createInitialMessages()
  const response1 = await runCompletion(inference, messages1)
  t.comment(`First completion (tools=true, len=${response1.length}):\n${response1}`)
  verifyReasoningTags(t, response1, 'First completion (tools=true)')

  // Second completion - this is where the fix should activate
  const messages2 = createFollowUpMessages(messages1, response1)
  const response2 = await runCompletion(inference, messages2)
  t.comment(`Second completion (tools=true, len=${response2.length}):\n${response2}`)

  verifyReasoningTags(t, response2, 'Second completion (tools=true)')

  // Verify the fix worked: generation continued after reasoning
  verifyContinuedAfterReasoning(t, response2, 'tools=true')
})

safeTest('Qwen3 reasoning-budget=0 disables thinking', {
  skip: isDarwinX64,
  timeout: 600_000
}, async t => {
  const [modelName, dirPath] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })
  const modelPath = path.join(dirPath, modelName)

  const baseConfig = {
    ctx_size: '4096',
    n_predict: '1024',
    seed: '50',
    gpu_layers: '999',
    temp: '0',
    top_p: '1',
    device: useCpu ? 'cpu' : 'gpu',
    verbosity: '0'
  }

  async function runOnce (extra) {
    const inference = new LlmLlamacpp({
      files: { model: [modelPath] },
      config: { ...baseConfig, ...extra },
      logger: console
    })
    try {
      await inference.load()
      const messages = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France? Answer in one word.' }
      ]
      return await runCompletion(inference, messages)
    } finally {
      await inference.unload().catch(() => {})
    }
  }

  const baseline = await runOnce({})
  const disabled = await runOnce({ 'reasoning-budget': '0' })
  const disabledUnderscore = await runOnce({ reasoning_budget: '0' })

  t.comment(`baseline (${baseline.length} chars): ${baseline.slice(0, 200)}`)
  t.comment(`disabled (${disabled.length} chars): ${disabled.slice(0, 200)}`)

  t.ok(/paris/i.test(baseline), 'baseline mentions Paris')
  t.ok(/paris/i.test(disabled), 'disabled mentions Paris')
  t.ok(/paris/i.test(disabledUnderscore), 'underscore variant also accepted and mentions Paris')

  // Baseline must show balanced reasoning markers in the stream. The Qwen3
  // template force-opens <think> in the prompt suffix; the addon prepends
  // the opener so streaming consumers see a matched <think>...</think> pair.
  t.ok(baseline.includes('<think>'),
    `baseline should contain <think> opening tag: "${baseline.slice(0, 100)}"`)
  t.ok(baseline.includes('</think>'),
    `baseline should contain </think> closing tag: "${baseline.slice(-100)}"`)
  t.ok(baseline.indexOf('<think>') < baseline.indexOf('</think>'),
    'baseline opening tag must precede closing tag')

  // With thinking disabled the visible stream skips the reasoning preamble
  // entirely, so neither marker should appear.
  t.absent(/<think>/.test(disabled),
    `disabled output should not contain <think>: "${disabled.slice(0, 200)}"`)
  t.absent(/<\/think>/.test(disabled),
    `disabled output should not contain </think>: "${disabled.slice(0, 200)}"`)
  t.ok(disabled.length < baseline.length / 4,
    `disabled (${disabled.length}) should be substantially shorter than baseline (${baseline.length})`)
})

// Default behaviour: without opting in, a Qwen3 turn that emits
// <think>...</think> should leave the thinking block in the cache and
// report 0 thinking-block discards. The opt-in path is covered by the
// next test, and the cross-turn effect by the multi-turn test below.
safeTest('remove_thinking_from_context defaults off for Qwen3', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(inference, messages)
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  verifyReasoningTags(t, response, 'default (no compaction)')

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 0,
    `default run should report 0 discards (got ${thinkingDiscards})`)
})

// Opt-in path: explicitly enabling the toggle drops the reasoning span
// from the KV cache. Mirrors the "defaults off" test but flips the flag.
safeTest('remove_thinking_from_context=true opts into compaction for Qwen3', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(
    inference,
    messages,
    { generationParams: { remove_thinking_from_context: true } }
  )
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  verifyReasoningTags(t, response, 'opt-in compaction')

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.ok(thinkingDiscards >= 1,
    `opt-in run should report at least one compaction (got ${thinkingDiscards})`)
})

// Opt-out path: when the caller explicitly disables the compaction, the
// runtime stats should report no discards and the cache should retain the
// full prompt + generated span (modulo the existing protected-first-message
// trimming the tools_compact controller already performs).
safeTest('remove_thinking_from_context=false keeps thinking in cache', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(
    inference,
    messages,
    { generationParams: { remove_thinking_from_context: false } }
  )
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  verifyReasoningTags(t, response, 'compaction disabled')

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 0,
    `compaction disabled should report 0 discards (got ${thinkingDiscards})`)
})

// Batch path opt-out: when the continuous-batching scheduler admits a
// request with `remove_thinking_from_context: false`, the per-slot driver
// must honour the toggle. Aggregated batch stats sum across slots, so a
// 0 here proves no slot dropped its thinking block.
safeTest('remove_thinking_from_context=false is honoured in batch path', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false, { configOverrides: { parallel: '2' } })

  const batchInput = [
    {
      id: 'q-france',
      prompt: createInitialMessages(),
      runOptions: { generationParams: { remove_thinking_from_context: false } }
    },
    {
      id: 'q-spain',
      prompt: [
        { role: 'system', content: 'You are an AI assistant. Always provide a clear answer after thinking' },
        { role: 'user', content: 'What is the capital of Spain?' }
      ],
      runOptions: { generationParams: { remove_thinking_from_context: false } }
    }
  ]

  const batchResponse = await inference.run(batchInput)
  const outputsById = new Map()
  await batchResponse
    .onUpdate(({ id, chunk }) => {
      outputsById.set(id, (outputsById.get(id) || '') + chunk)
    })
    .await()
  const stats = batchResponse.stats || {}
  t.comment(`batch stats: ${JSON.stringify(stats)}`)

  for (const item of batchInput) {
    const output = outputsById.get(item.id) || ''
    t.comment(`batch ${item.id} (len=${output.length}): ${output.slice(0, 160)}...`)
    t.ok(output.includes('<think>') && output.includes('</think>'),
      `batch ${item.id} should retain <think>...</think> tags`)
  }

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 0,
    `batch path with compaction disabled should report 0 discards (got ${thinkingDiscards})`)
})

// Mixed-slot batch path: per-slot drivers honour their own
// `remove_thinking_from_context` overrides independently. Slot A opts in
// (1 discard), slot B leaves the toggle at its default-off (0 discards);
// the scheduler's `accumulateSlotRuntimeStats` sums per-slot
// `getThinkingBlockDiscards()` so the aggregate must be exactly 1.
safeTest('batch path aggregates per-slot remove_thinking_from_context independently', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false, { configOverrides: { parallel: '2' } })

  const batchInput = [
    {
      id: 'slot-on',
      prompt: createInitialMessages(),
      runOptions: { generationParams: { remove_thinking_from_context: true } }
    },
    {
      id: 'slot-off',
      prompt: [
        { role: 'system', content: 'You are an AI assistant. Always provide a clear answer after thinking' },
        { role: 'user', content: 'What is the capital of Spain?' }
      ]
      // No runOptions → compaction stays at its default-off for this slot.
    }
  ]

  const batchResponse = await inference.run(batchInput)
  const outputsById = new Map()
  await batchResponse
    .onUpdate(({ id, chunk }) => {
      outputsById.set(id, (outputsById.get(id) || '') + chunk)
    })
    .await()
  const stats = batchResponse.stats || {}
  t.comment(`mixed-slot batch stats: ${JSON.stringify(stats)}`)

  for (const item of batchInput) {
    const output = outputsById.get(item.id) || ''
    t.comment(`mixed-slot ${item.id} (len=${output.length}): ${output.slice(0, 160)}...`)
    t.ok(output.includes('<think>') && output.includes('</think>'),
      `mixed-slot ${item.id} output should contain <think>...</think>`)
  }

  // Slot A (opt-in) contributes 1; slot B (default-off) contributes 0.
  // Sum across slots must equal 1 — proves per-slot independence AND
  // that `accumulateSlot` actually sums the per-slot value (not max / overwrite).
  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 1,
    'mixed-slot batch should aggregate to exactly 1 discard ' +
    `(slot-on=1, slot-off=0), got ${thinkingDiscards}`)
})

// reasoning_budget=0 short-circuits the channel before any tokens are
// emitted, so the compaction feature has nothing to do and reports 0
// discards even when `remove_thinking_from_context: true` is opted in.
safeTest('remove_thinking_from_context is a no-op when reasoning_budget=0', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(
    inference,
    messages,
    {
      generationParams: {
        reasoning_budget: 0,
        remove_thinking_from_context: true
      }
    }
  )
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 0,
    `reasoning_budget=0 should report 0 discards (got ${thinkingDiscards})`)
  t.absent(/<think>/.test(response),
    `reasoning_budget=0 output should not contain <think>: "${response.slice(0, 200)}"`)
})

// Multi-turn cache growth comparison. Uses a `cacheKey` so the KV cache
// persists across `run()` calls (without it the addon resets `nPast_` to 0
// after every inference and the cross-turn effect is invisible). Runs the
// same two-turn flow twice: once with compaction explicitly opted in and
// once with compaction off; the off run should have a larger residual cache.
safeTest('remove_thinking_from_context reduces multi-turn cache growth', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 1_200_000
}, async t => {
  const sessionA = path.join(os.tmpdir(), `qvac-think-compact-on-${Date.now()}.bin`)
  const sessionB = path.join(os.tmpdir(), `qvac-think-compact-off-${Date.now() + 1}.bin`)

  t.teardown(() => {
    for (const p of [sessionA, sessionB]) {
      try { require('bare-fs').unlinkSync(p) } catch {}
    }
  })

  const messages1 = createInitialMessages()
  const overridesOn = { generationParams: { remove_thinking_from_context: true } }

  // Run A — compaction ON (explicit opt-in).
  const { inference: infA } = await setupReasoningModel(t, false)
  const a1 = await runCompletionWithStats(infA, messages1, { cacheKey: sessionA, ...overridesOn })
  verifyReasoningTags(t, a1.response, 'A turn 1')
  t.ok(toNumber(a1.stats.thinkingBlockDiscards) >= 1,
    'A turn 1 should compact at least one thinking block')
  const a2 = await runCompletionWithStats(
    infA,
    createFollowUpMessages(messages1, a1.response),
    { cacheKey: sessionA, ...overridesOn }
  )
  verifyReasoningTags(t, a2.response, 'A turn 2')
  // Symmetric guard on turn 2: the cross-turn delta below assumes BOTH
  // turns of run A produced and compacted a thinking block. Without this
  // guard, a turn-2 that silently skipped thinking would still pass the
  // `cacheA2 < cacheB2` assertion (turn-1 delta alone is enough), but the
  // test would have lost half its discriminating power.
  t.ok(toNumber(a2.stats.thinkingBlockDiscards) >= 1,
    'A turn 2 should also compact at least one thinking block')

  // Run B — same flow, compaction OFF.
  const { inference: infB } = await setupReasoningModel(t, false)
  const overridesOff = { generationParams: { remove_thinking_from_context: false } }
  const b1 = await runCompletionWithStats(
    infB,
    messages1,
    { cacheKey: sessionB, ...overridesOff }
  )
  verifyReasoningTags(t, b1.response, 'B turn 1')
  t.is(toNumber(b1.stats.thinkingBlockDiscards), 0,
    'B turn 1 with compaction off should report 0 discards')
  const b2 = await runCompletionWithStats(
    infB,
    createFollowUpMessages(messages1, b1.response),
    { cacheKey: sessionB, ...overridesOff }
  )
  verifyReasoningTags(t, b2.response, 'B turn 2')

  const cacheA2 = toNumber(a2.stats.CacheTokens)
  const cacheB2 = toNumber(b2.stats.CacheTokens)
  t.comment(`compaction ON  turn 2 cache=${cacheA2} stats=${JSON.stringify(a2.stats)}`)
  t.comment(`compaction OFF turn 2 cache=${cacheB2} stats=${JSON.stringify(b2.stats)}`)

  t.ok(cacheA2 > 0, `compaction-on turn 2 should have non-zero cache (got ${cacheA2})`)
  t.ok(cacheB2 > 0, `compaction-off turn 2 should have non-zero cache (got ${cacheB2})`)
  t.ok(cacheA2 < cacheB2,
    `turn 2 cache with compaction ON (${cacheA2}) should be < OFF (${cacheB2}) — proves turn 1 thinking was dropped from the cache`)
})

// Qwen3.5 coverage — exercises the reasoning detection on a hybrid SSM
// checkpoint and verifies the recurrent-memory gate keeps the cache
// untouched. Qwen3.5 thinking traces can exceed 1k tokens before
// `</think>` closes, so we give a larger n_predict / ctx_size.
const QWEN35_REASONING_CONFIG = {
  ctx_size: '8192',
  n_predict: '3072'
}

// Qwen3.5 is a hybrid SSM family. The recurrent half is rolled back
// via a partial-only `llama_state_seq_get_data_ext` snapshot taken at
// the open marker, restored at end-of-generation, and the post-
// reasoning tail is replayed through `llama_decode` so the SSM advances
// over it without absorbing the dropped span. The previous hard
// rejection has been removed; this test pins the success path.
safeTest('Qwen3.5 honours remove_thinking_from_context opt-in', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 900_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false, {
    modelDef: QWEN35_MODEL,
    configOverrides: QWEN35_REASONING_CONFIG
  })

  const messages = createInitialMessages()

  const { response, stats } = await runCompletionWithStats(
    inference,
    messages,
    { generationParams: { remove_thinking_from_context: true } }
  )
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  // The model produced visible reasoning tags during generation — the
  // compactor only drops a span if `<think>...</think>` actually fired.
  verifyReasoningTags(t, response, 'Qwen3.5 opt-in')

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  const compactionFailed = toNumber(stats.thinkingCompactionFailed)
  t.is(compactionFailed, 0,
    `recurrent restore + replay should succeed (got ${compactionFailed} failures)`)
  t.ok(thinkingDiscards >= 1,
    `opt-in run should report at least one discard (got ${thinkingDiscards})`)
})

// Multi-turn assertion that the SSM rollback is doing its job: with
// compaction ON, turn-2 must not be measurably steered by turn-1's
// reasoning span. We measure indirectly via cache growth — turn-2's
// pre-decode `cacheTokens` should equal the protected-prefix tokens
// plus turn-2's own prompt, NOT the turn-1 reasoning body. A regression
// where the snapshot/replay was wired up incorrectly (e.g. forgetting
// to drop the attention KV) would leave turn-2 carrying turn-1's
// thinking and the assertion below would fail.
safeTest('Qwen3.5 multi-turn with remove_thinking_from_context is reasoning-clean', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 1_500_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false, {
    modelDef: QWEN35_MODEL,
    configOverrides: QWEN35_REASONING_CONFIG
  })

  const messagesT1 = createInitialMessages()

  const t1 = await runCompletionWithStats(
    inference,
    messagesT1,
    { generationParams: { remove_thinking_from_context: true } }
  )
  t.comment(`turn 1 stats: ${JSON.stringify(t1.stats)}`)
  t.is(toNumber(t1.stats.thinkingCompactionFailed), 0,
    'turn 1 compaction should not fail')
  t.ok(toNumber(t1.stats.thinkingBlockDiscards) >= 1,
    'turn 1 should drop at least one reasoning block')

  const messagesT2 = [
    ...messagesT1,
    { role: 'assistant', content: t1.response },
    { role: 'user', content: 'Now tell me the capital of Spain.' }
  ]

  const t2 = await runCompletionWithStats(
    inference,
    messagesT2,
    { generationParams: { remove_thinking_from_context: true } }
  )
  t.comment(`turn 2 stats: ${JSON.stringify(t2.stats)}`)
  t.comment(`turn 2 response (len=${t2.response.length}): ${t2.response.slice(0, 300)}`)
  t.is(toNumber(t2.stats.thinkingCompactionFailed), 0,
    'turn 2 compaction should not fail')
  t.ok(t2.response.length > 0,
    'turn 2 should still produce a response (generation succeeds after rollback)')

  // Functional check on the answer itself. The end-of-prefill snapshot
  // keeps the forced `<think>\n` opener in the SSM hidden state; if
  // the replay buffer omits the matching close marker, turn 2 inherits
  // an unbalanced (opener-without-closer) recurrent state and the
  // resulting answer tends to drift off-topic or loop. With temp=0,
  // a balanced replay reliably produces "Madrid" somewhere in the
  // response. A degenerate SSM does not.
  t.ok(/madrid/i.test(t2.response),
    'turn 2 should answer "capital of Spain" with Madrid (proves the SSM did not degenerate)')
})

// `runtimeStats()` reports a per-inference user-visible perf snapshot
// captured at the start of `compactThinkSpan`. On hybrid SSM models
// the compactor then runs `restore + llama_decode` to replay the post-
// reasoning tail through the SSM; without the snapshot, those replay
// decodes accumulate into `n_p_eval` / `t_p_eval_ms` and inflate
// user-facing `promptTokens` (and `ppTPS` / `TTFT`) by the replay
// length. This regression test pins the contract by running the same
// prompt + seed twice on Qwen3.5 with compaction toggled. Both runs
// share the same prefill, so a non-inflated `promptTokens` must match
// to within the noise floor introduced by per-instance load
// determinism — the `=false` baseline gives the true prefill count
// without any replay path. Without the snapshot the `=true` run is
// strictly larger; with the snapshot the two runs report the same
// `promptTokens`.
safeTest('Qwen3.5 remove_thinking_from_context does not inflate runtime perf stats', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 1_800_000
}, async t => {
  const [modelName, dirPath] = await ensureModel({
    modelName: QWEN35_MODEL.name,
    downloadUrl: QWEN35_MODEL.url
  })
  const modelPath = path.join(dirPath, modelName)

  const baseConfig = {
    device: useCpu ? 'cpu' : 'gpu',
    gpu_layers: '999',
    seed: '50',
    temp: '0',
    top_p: '1',
    verbosity: '2',
    ...QWEN35_REASONING_CONFIG
  }

  async function runOnce (removeThinking) {
    const inference = new LlmLlamacpp({
      files: { model: [modelPath] },
      config: baseConfig,
      logger: console,
      opts: { stats: true }
    })
    try {
      await inference.load()
      const messages = createInitialMessages()
      const { stats } = await runCompletionWithStats(
        inference,
        messages,
        { generationParams: { remove_thinking_from_context: removeThinking } }
      )
      return stats
    } finally {
      await inference.unload().catch(() => {})
    }
  }

  // Baseline first: compaction off, no replay decode, perf counters
  // reflect a clean prefill.
  const off = await runOnce(false)
  t.comment(`compaction=off stats: ${JSON.stringify(off)}`)

  // Then with compaction on. Same prompt + seed + cfg, so the prefill
  // token count is byte-for-byte identical. The only difference is
  // that the hybrid replay decode runs after generation.
  const on = await runOnce(true)
  t.comment(`compaction=on  stats: ${JSON.stringify(on)}`)

  t.is(toNumber(on.thinkingCompactionFailed), 0,
    'compaction-on run must not fail (otherwise the snapshot-and-replay path was not exercised)')
  t.ok(toNumber(on.thinkingBlockDiscards) >= 1,
    'compaction-on run must actually drop a reasoning block (otherwise no replay decode ran)')

  // The contract: `promptTokens` reflects the user-visible prefill,
  // NOT the prefill plus the replayed post-reasoning tail. With the
  // snapshot fix the two runs match; without it the compaction-on run
  // is strictly larger by the replay length.
  t.is(toNumber(on.promptTokens), toNumber(off.promptTokens),
    `promptTokens must match between compaction on/off (on=${on.promptTokens}, off=${off.promptTokens}); ` +
    'a larger on-value means the recurrent replay decode was counted as user-visible prompt work')
})

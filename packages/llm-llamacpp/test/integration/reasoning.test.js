'use strict'

const path = require('bare-path')
const { ensureModel, safeTest } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const os = require('bare-os')
const LlmLlamacpp = require('../../index.js')

const isDarwinX64 = os.platform() === 'darwin' && os.arch() === 'x64'
const isLinuxArm64 = os.platform() === 'linux' && os.arch() === 'arm64'
const useCpu = isLinuxArm64

const MODEL = {
  name: 'Qwen3-0.6B-Q8_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf'
}

async function setupReasoningModel (t, toolsEnabled) {
  const [modelName, dirPath] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
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
    tools: toolsEnabled ? 'true' : 'false'
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

// Compaction-on-by-default: after a Qwen3 turn that emits <think>...</think>,
// the runtime stats should report at least one thinking block discarded, and
// the cache after the turn should be smaller than the naive sum of prompt +
// generated tokens (since the thinking span was dropped).
safeTest('remove_thinking_from_context defaults on for Qwen3', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(inference, messages)
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  verifyReasoningTags(t, response, 'default compaction')

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.ok(thinkingDiscards >= 1,
    `default run should report at least one compaction (got ${thinkingDiscards})`)

  // CacheTokens should be smaller than prompt + generated: the discarded
  // thinking span is no longer accounted for in the residual cache.
  const cacheTokens = toNumber(stats.CacheTokens)
  const promptTokens = toNumber(stats.promptTokens)
  const generatedTokens = toNumber(stats.generatedTokens)
  t.ok(cacheTokens > 0, `cacheTokens should be positive (got ${cacheTokens})`)
  t.ok(cacheTokens < promptTokens + generatedTokens,
    `cacheTokens (${cacheTokens}) should be < prompt (${promptTokens}) + generated (${generatedTokens}) when thinking was compacted out`)
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

// reasoning_budget=0 short-circuits the channel before any tokens are
// emitted, so the compaction feature has nothing to do and reports 0
// discards even with the default `remove_thinking_from_context: true`.
safeTest('remove_thinking_from_context is a no-op when reasoning_budget=0', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 600_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages = createInitialMessages()
  const { response, stats } = await runCompletionWithStats(
    inference,
    messages,
    { generationParams: { reasoning_budget: 0 } }
  )
  t.comment(`response (len=${response.length}): ${response.slice(0, 200)}...`)
  t.comment(`stats: ${JSON.stringify(stats)}`)

  const thinkingDiscards = toNumber(stats.thinkingBlockDiscards)
  t.is(thinkingDiscards, 0,
    `reasoning_budget=0 should report 0 discards (got ${thinkingDiscards})`)
  t.absent(/<think>/.test(response),
    `reasoning_budget=0 output should not contain <think>: "${response.slice(0, 200)}"`)
})

// Multi-turn cache growth comparison. With compaction enabled (default), the
// turn-2 CacheTokens should be smaller than what we would see if the turn-1
// thinking block remained in the cache. We anchor the assertion against the
// turn-1 stats so the test is independent of the model's specific verbosity.
safeTest('remove_thinking_from_context reduces multi-turn cache growth', {
  skip: isDarwinX64 || isWindowsX64,
  timeout: 900_000
}, async t => {
  const { inference } = await setupReasoningModel(t, false)

  const messages1 = createInitialMessages()
  const { response: response1, stats: stats1 } = await runCompletionWithStats(
    inference,
    messages1
  )
  verifyReasoningTags(t, response1, 'turn 1')
  t.ok(toNumber(stats1.thinkingBlockDiscards) >= 1,
    'turn 1 should compact at least one thinking block')

  const messages2 = createFollowUpMessages(messages1, response1)
  const { response: response2, stats: stats2 } = await runCompletionWithStats(
    inference,
    messages2
  )
  verifyReasoningTags(t, response2, 'turn 2')

  const cache1 = toNumber(stats1.CacheTokens)
  const cache2 = toNumber(stats2.CacheTokens)
  const promptTokens2 = toNumber(stats2.promptTokens)
  const generatedTokens2 = toNumber(stats2.generatedTokens)

  t.comment(`turn1: cache=${cache1} stats=${JSON.stringify(stats1)}`)
  t.comment(`turn2: cache=${cache2} stats=${JSON.stringify(stats2)}`)

  // Even after adding turn 2's prompt + generated tokens, the new cache
  // should be less than a naive accumulation that would include turn 1's
  // thinking block (we compacted it).
  const naive = cache1 + promptTokens2 + generatedTokens2
  t.ok(cache2 < naive,
    `turn 2 cache (${cache2}) should be < naive accumulation (${naive}) — proves turn 1 thinking was compacted out`)
})

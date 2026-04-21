'use strict'

const test = require('brittle')
const LlmLlamacpp = require('../../index.js')
const { ensureModel } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const path = require('bare-path')

const MODEL = {
  name: 'Qwen3-0.6B-Q8_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf'
}

const PROMPT = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of France? Answer in one word.' }
]

function extractBufferDevices (logs) {
  const deviceNames = new Set()
  for (const line of logs) {
    const match = line.match(/(\S+)\s+model buffer size\s*=/)
    if (match) deviceNames.add(match[1])
  }
  return deviceNames
}

async function collectResponse (response) {
  const chunks = []
  await response.onUpdate(data => { chunks.push(data) }).await()
  return chunks.join('').trim()
}

const gpuCount = LlmLlamacpp.getGpuDeviceCount()
const hasMultiGpu = gpuCount >= 2

test('multi-gpu: split-mode=layer distributes layers across GPUs', { timeout: 600_000 }, async t => {
  if (!hasMultiGpu) {
    t.comment(`Skipping: detected ${gpuCount} GPU(s), requires at least 2`)
    return
  }

  const [modelName, dirPath] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })

  const modelPath = path.join(dirPath, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const addon = new LlmLlamacpp({
    files: { model: [modelPath] },
    config: {
      device: 'gpu',
      gpu_layers: '999',
      ctx_size: '1024',
      n_predict: '32',
      'split-mode': 'layer',
      verbosity: '2'
    },
    logger: null,
    opts: { stats: true }
  })

  try {
    await addon.load()
    const response = await addon.run(PROMPT)
    const output = await collectResponse(response)
    const stats = response.stats || {}

    t.ok(output.length > 0, 'should generate output')
    t.is(stats.backendDevice, 'gpu', 'should report gpu backend')

    const devices = extractBufferDevices(specLogger.logs)
    t.ok(devices.size >= 2, `layers should be on >= 2 devices (found: ${[...devices].join(', ')})`)
  } finally {
    specLogger.release()
    await addon.unload().catch(() => {})
  }
})

test('multi-gpu: split-mode=row distributes tensors across GPUs', { timeout: 600_000 }, async t => {
  if (!hasMultiGpu) {
    t.comment(`Skipping: detected ${gpuCount} GPU(s), requires at least 2`)
    return
  }

  const [modelName, dirPath] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })

  const modelPath = path.join(dirPath, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const addon = new LlmLlamacpp({
    files: { model: [modelPath] },
    config: {
      device: 'gpu',
      gpu_layers: '999',
      ctx_size: '1024',
      n_predict: '32',
      'split-mode': 'row',
      verbosity: '2'
    },
    logger: null,
    opts: { stats: true }
  })

  try {
    await addon.load()
    const response = await addon.run(PROMPT)
    const output = await collectResponse(response)
    const stats = response.stats || {}

    t.ok(output.length > 0, 'should generate output')
    t.is(stats.backendDevice, 'gpu', 'should report gpu backend')

    const devices = extractBufferDevices(specLogger.logs)
    t.ok(devices.size >= 2, `tensors should be on >= 2 devices (found: ${[...devices].join(', ')})`)
  } finally {
    specLogger.release()
    await addon.unload().catch(() => {})
  }
})

test('multi-gpu: split-mode=layer with tensor-split and main-gpu', { timeout: 600_000 }, async t => {
  if (!hasMultiGpu) {
    t.comment(`Skipping: detected ${gpuCount} GPU(s), requires at least 2`)
    return
  }

  const [modelName, dirPath] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })

  const modelPath = path.join(dirPath, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const addon = new LlmLlamacpp({
    files: { model: [modelPath] },
    config: {
      device: 'gpu',
      gpu_layers: '999',
      ctx_size: '1024',
      n_predict: '32',
      'split-mode': 'layer',
      'tensor-split': '1,1',
      'main-gpu': '0',
      verbosity: '2'
    },
    logger: null,
    opts: { stats: true }
  })

  try {
    await addon.load()
    const response = await addon.run(PROMPT)
    const output = await collectResponse(response)
    const stats = response.stats || {}

    t.ok(output.length > 0, 'should generate output')
    t.is(stats.backendDevice, 'gpu', 'should report gpu backend')

    const devices = extractBufferDevices(specLogger.logs)
    t.ok(devices.size >= 2, `layers should be on >= 2 devices (found: ${[...devices].join(', ')})`)
  } finally {
    specLogger.release()
    await addon.unload().catch(() => {})
  }
})

setImmediate(() => {
  setTimeout(() => {}, 500)
})

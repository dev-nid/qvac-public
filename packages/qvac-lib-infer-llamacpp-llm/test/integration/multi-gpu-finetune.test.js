'use strict'

const test = require('brittle')
const process = require('bare-process')
const path = require('bare-path')
const LlmLlamacpp = require('../../index.js')
const { ensureModel, setupParams, cleanupCheckpoints } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')

const MODEL = {
  name: 'Qwen3-0.6B-Q8_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf'
}

const hasMultiGpu = process.env.QVAC_HAS_MULTI_GPU === '1'

function extractBufferDevices (logs) {
  const deviceNames = new Set()
  for (const line of logs) {
    const match = line.match(/\b((?:Vulkan|CUDA|Metal|ROCm|SYCL|OpenCL)\d*)\b\s+model buffer size\s*=/i)
    if (match) deviceNames.add(match[1])
  }
  return deviceNames
}

async function runMultiGpuFinetuneTest (t, extraConfig, assertDevices) {
  if (!hasMultiGpu) {
    t.comment('Skipping: QVAC_HAS_MULTI_GPU is not set')
    return
  }

  const [modelName, modelDir] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })

  const modelPath = path.join(modelDir, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const finetuneConfig = setupParams(modelDir, {
    checkpointSaveSteps: 0,
    datasetSize: 16,
    testId: 'multi-gpu-ft'
  })
  const checkpointDir = finetuneConfig.checkpointSaveDir

  const config = {
    device: 'gpu',
    gpu_layers: '999',
    ctx_size: '512',
    verbosity: '2',
    ...extraConfig
  }

  const addon = new LlmLlamacpp({
    files: { model: [modelPath] },
    config,
    logger: null,
    opts: { stats: true }
  })

  try {
    await addon.load()

    const finetuneHandle = await addon.finetune(finetuneConfig)

    let progressCount = 0
    finetuneHandle.on('stats', stats => {
      progressCount++
      t.comment(`progress: epoch=${stats.current_epoch + 1} step=${stats.global_steps} loss=${stats.loss?.toFixed(4)}`)
    })

    const result = await finetuneHandle.await()

    t.ok(result, 'finetune should return a result')
    t.ok(progressCount > 0, 'should receive progress events')
    t.ok(
      result.status === 'COMPLETED',
      `finetune should complete, got: ${result.status}`
    )

    const devices = extractBufferDevices(specLogger.logs)
    assertDevices(t, devices)
  } finally {
    specLogger.release()
    await addon.unload().catch(() => {})
    cleanupCheckpoints(checkpointDir)
  }
}

function assertMultiDevice (label) {
  return (t, devices) => {
    t.ok(devices.size >= 2, `${label} should be on >= 2 devices (found: ${[...devices].join(', ')})`)
  }
}

function assertSingleDevice (t, devices) {
  t.ok(devices.size <= 1, `finetuning layers should stay on a single device (found: ${[...devices].join(', ')})`)
}

test('multi-gpu finetune: split-mode=layer distributes across GPUs', { timeout: 600_000 }, async t => {
  await runMultiGpuFinetuneTest(t, { 'split-mode': 'layer' }, assertMultiDevice('finetune layers'))
})

test('multi-gpu finetune: split-mode=row distributes tensors across GPUs', { timeout: 600_000 }, async t => {
  await runMultiGpuFinetuneTest(t, { 'split-mode': 'row' }, assertMultiDevice('finetune tensors'))
})

test('multi-gpu finetune: default (no split-mode) uses single device', { timeout: 600_000 }, async t => {
  await runMultiGpuFinetuneTest(t, {}, assertSingleDevice)
})

test('multi-gpu finetune: split-mode=layer with tensor-split weighting', { timeout: 600_000 }, async t => {
  await runMultiGpuFinetuneTest(
    t,
    { 'split-mode': 'layer', 'tensor-split': '1,1', 'main-gpu': '0' },
    assertMultiDevice('finetune weighted layers')
  )
})

test('multi-gpu finetune: inference after multi-gpu finetuning produces output', { timeout: 600_000 }, async t => {
  if (!hasMultiGpu) {
    t.comment('Skipping: QVAC_HAS_MULTI_GPU is not set')
    return
  }

  const [modelName, modelDir] = await ensureModel({
    modelName: MODEL.name,
    downloadUrl: MODEL.url
  })

  const modelPath = path.join(modelDir, modelName)
  const specLogger = attachSpecLogger({ forwardToConsole: true })

  const finetuneConfig = setupParams(modelDir, {
    checkpointSaveSteps: 0,
    datasetSize: 16,
    testId: 'multi-gpu-ft-infer'
  })
  const checkpointDir = finetuneConfig.checkpointSaveDir

  const config = {
    device: 'gpu',
    gpu_layers: '999',
    ctx_size: '512',
    verbosity: '2',
    'split-mode': 'layer'
  }

  const addon = new LlmLlamacpp({
    files: { model: [modelPath] },
    config,
    logger: null,
    opts: { stats: true }
  })

  try {
    await addon.load()

    const finetuneHandle = await addon.finetune(finetuneConfig)
    const result = await finetuneHandle.await()
    t.ok(result.status === 'COMPLETED', `finetune should complete, got: ${result.status}`)

    const loraAdapterPath = path.join(finetuneConfig.outputParametersDir, 'trained-lora-adapter.gguf')

    const inferConfig = {
      device: 'gpu',
      gpu_layers: '999',
      ctx_size: '512',
      n_predict: '32',
      verbosity: '2',
      'split-mode': 'layer',
      lora: loraAdapterPath
    }

    const inferAddon = new LlmLlamacpp({
      files: { model: [modelPath] },
      config: inferConfig,
      logger: null,
      opts: { stats: true }
    })

    await inferAddon.load()

    const prompt = [
      { role: 'user', content: 'Hello' }
    ]
    const response = await inferAddon.run(prompt)
    let generated = ''
    await response.onUpdate(token => { generated += token }).await()

    t.ok(generated.length > 0, 'inference with finetuned LoRA on multi-GPU should produce output')
    t.comment(`Generated (${generated.length} chars): ${generated.slice(0, 100)}`)

    const devices = extractBufferDevices(specLogger.logs)
    t.ok(devices.size >= 2, `inference should use multiple devices (found: ${[...devices].join(', ')})`)

    await inferAddon.unload().catch(() => {})
  } finally {
    specLogger.release()
    await addon.unload().catch(() => {})
    cleanupCheckpoints(checkpointDir)
  }
})

setImmediate(() => {
  setTimeout(() => {}, 500)
})

'use strict'

const LlamaClient = require('../../index')
const process = require('bare-process')
const path = require('bare-path')
const fs = require('bare-fs')
const { downloadModel, formatProgress, createFilteredLogger } = require('../utils')

// ──────────────────────────────────────────────────────────────────────────────
// Model — same large model for both sections so timings are comparable
// ──────────────────────────────────────────────────────────────────────────────

const MODEL = {
  name: 'Qwen3-32B-Q4_0.gguf',
  url: 'https://huggingface.co/unsloth/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_0.gguf'
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI helpers
// ──────────────────────────────────────────────────────────────────────────────

function parseIntegerArg (name, defaultValue) {
  const arg = process.argv.find(a => a.startsWith(`--${name}=`))
  if (!arg) return defaultValue
  const value = Number.parseInt(arg.split('=')[1], 10)
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid --${name} value`)
  }
  return value
}

function parseStringArg (name, defaultValue) {
  const arg = process.argv.find(a => a.startsWith(`--${name}=`))
  if (!arg) return defaultValue
  return arg.slice(`--${name}=`.length)
}

function hasFlag (name) {
  return process.argv.includes(`--${name}`)
}

// ──────────────────────────────────────────────────────────────────────────────
// Formatting / progress
// ──────────────────────────────────────────────────────────────────────────────

function fmt (value, digits) {
  digits = digits || 2
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a'
}

function waitForProgress (handle, minSteps, timeoutMs) {
  minSteps = minSteps || 5
  timeoutMs = timeoutMs || 600_000
  return new Promise((resolve, reject) => {
    let count = 0
    const timer = setTimeout(() => {
      handle.removeListener('stats', onStats)
      reject(new Error(`waitForProgress: timed out after ${timeoutMs}ms (${count}/${minSteps} steps)`))
    }, timeoutMs)
    const onStats = () => {
      if (++count >= minSteps) {
        clearTimeout(timer)
        handle.removeListener('stats', onStats)
        resolve()
      }
    }
    handle.on('stats', onStats)
  })
}

const PAUSE_CHECKPOINT_PREFIX = 'pause_checkpoint_step_'

function listPauseCheckpointDirs (checkpointDir) {
  if (!fs.existsSync(checkpointDir)) return []
  const entries = fs.readdirSync(checkpointDir, { withFileTypes: true })
  return entries
    .filter(e => e.isDirectory() && e.name.startsWith(PAUSE_CHECKPOINT_PREFIX))
    .map(e => ({ name: e.name, step: parseInt(e.name.slice(PAUSE_CHECKPOINT_PREFIX.length), 10) }))
    .filter(p => !isNaN(p.step))
}

function latestPauseCheckpointPath (checkpointDir) {
  const dirs = listPauseCheckpointDirs(checkpointDir)
  if (dirs.length === 0) return null
  const latest = dirs.reduce((a, b) => (a.step > b.step ? a : b))
  return path.join(checkpointDir, latest.name)
}

function cleanupDir (dirPath) {
  if (fs.existsSync(dirPath)) {
    try { fs.rmSync(dirPath, { recursive: true, force: true }) } catch (_) {}
  }
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

// ──────────────────────────────────────────────────────────────────────────────
// Section 1: Single-GPU pause/resume finetuning (small model)
// ──────────────────────────────────────────────────────────────────────────────

async function runSingleGpuPauseResume (logger, opts) {
  const epochs = opts.epochs

  console.log('\n' + '='.repeat(72))
  console.log('SECTION 1: Single-GPU Finetune with Pause/Resume (Qwen3-32B-Q4_0)')
  console.log('='.repeat(72) + '\n')

  const [modelName, modelDir] = await downloadModel(MODEL.url, MODEL.name)
  const modelPath = path.join(modelDir, modelName)

  const checkpointDir = './benchmark_checkpoints/single-gpu'
  const outputDir = './benchmark_output/single-gpu'
  cleanupDir(checkpointDir)
  cleanupDir(outputDir)

  const config = {
    device: 'gpu',
    gpu_layers: '999',
    ctx_size: '512',
    verbosity: '2'
  }

  const finetuneOptions = {
    trainDatasetDir: './examples/input/small_train_HF.jsonl',
    validation: { type: 'dataset', path: './examples/input/small_eval_HF.jsonl' },
    numberOfEpochs: epochs,
    learningRate: 1e-5,
    lrMin: 1e-8,
    batchSize: 32,
    microBatchSize: 8,
    loraModules: 'attn_q,attn_k,attn_v,attn_o',
    assistantLossOnly: true,
    checkpointSaveSteps: 10,
    checkpointSaveDir: checkpointDir,
    outputParametersDir: outputDir
  }

  const client = new LlamaClient({
    files: { model: [modelPath] },
    config,
    logger,
    opts: { stats: true }
  })

  try {
    console.log('Loading model...')
    const loadStart = Date.now()
    await client.load()
    const loadTimeMs = Date.now() - loadStart
    console.log(`Model loaded in ${loadTimeMs}ms\n`)

    // ── Phase 1: finetune then pause ──
    console.log('Starting finetuning (will pause after 5 steps)...')
    const startTime = Date.now()
    const handle = await client.finetune(finetuneOptions)
    handle.on('stats', stats => {
      console.log(`  ${formatProgress(stats, finetuneOptions.numberOfEpochs)}`)
    })

    await waitForProgress(handle, 5)
    console.log('\nPausing finetuning...')
    await client.pause()
    const pauseResult = await handle.await()
    const pauseElapsed = Date.now() - startTime
    console.log(`Pause result: ${JSON.stringify(pauseResult)}`)
    console.log(`Time to pause: ${pauseElapsed}ms\n`)

    if (pauseResult?.status === 'COMPLETED') {
      console.log('Training completed before pause could take effect.')
      console.log(`Total time: ${pauseElapsed}ms`)
      return { totalTimeMs: pauseElapsed, status: 'COMPLETED', pauseTimeMs: pauseElapsed, resumeTimeMs: 0 }
    }

    // Verify checkpoint
    for (let retry = 0; retry < 10; retry++) {
      const pausePath = latestPauseCheckpointPath(checkpointDir)
      if (pausePath && fs.existsSync(path.join(pausePath, 'metadata.txt'))) {
        console.log(`Pause checkpoint: ${pausePath}`)
        break
      }
      await sleep(500)
    }

    // ── Phase 2: resume ──
    console.log('\nResuming finetuning...')
    const resumeStart = Date.now()
    const resumeHandle = await client.finetune(finetuneOptions)
    resumeHandle.on('stats', stats => {
      console.log(`  ${formatProgress(stats, finetuneOptions.numberOfEpochs)}`)
    })
    const resumeResult = await resumeHandle.await()
    const resumeElapsed = Date.now() - resumeStart
    const totalElapsed = Date.now() - startTime

    console.log(`\nResume result: ${JSON.stringify(resumeResult)}`)
    console.log(`Resume time: ${resumeElapsed}ms`)
    console.log(`Total time (pause + resume): ${totalElapsed}ms`)

    return {
      totalTimeMs: totalElapsed,
      pauseTimeMs: pauseElapsed,
      resumeTimeMs: resumeElapsed,
      status: resumeResult?.status || 'UNKNOWN',
      stats: resumeResult?.stats
    }
  } finally {
    await client.unload().catch(() => {})
    cleanupDir(checkpointDir)
    cleanupDir(outputDir)
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Section 2: Multi-GPU finetune benchmark (large model)
// ──────────────────────────────────────────────────────────────────────────────

async function runLargeModelFinetuneBenchmark (logger, opts) {
  const tensorSplit = opts.tensorSplit
  const epochs = opts.epochs

  console.log('\n' + '='.repeat(72))
  console.log('SECTION 2: Large-Model Finetune Benchmark (Qwen3-32B-Q4_0)')
  console.log('Comparing: single GPU vs layer-split vs row-split')
  console.log('='.repeat(72) + '\n')

  const [modelName, modelDir] = await downloadModel(MODEL.url, MODEL.name)
  const modelPath = path.join(modelDir, modelName)

  const finetuneOptionsBase = {
    trainDatasetDir: './examples/input/small_train_HF.jsonl',
    validation: { type: 'dataset', path: './examples/input/small_eval_HF.jsonl' },
    numberOfEpochs: epochs,
    learningRate: 1e-5,
    lrMin: 1e-8,
    batchSize: 32,
    microBatchSize: 8,
    loraModules: 'attn_q,attn_k,attn_v,attn_o',
    assistantLossOnly: true,
    checkpointSaveSteps: 0
  }

  const modes = [
    {
      label: 'Single GPU',
      config: { device: 'gpu', gpu_layers: '999', ctx_size: '512', verbosity: '2' }
    },
    {
      label: 'Multi-GPU layer-split',
      config: { device: 'gpu', gpu_layers: '999', ctx_size: '512', verbosity: '2', 'split-mode': 'layer', 'tensor-split': tensorSplit }
    },
    {
      label: 'Multi-GPU row-split',
      config: { device: 'gpu', gpu_layers: '999', ctx_size: '512', verbosity: '2', 'split-mode': 'row', 'tensor-split': tensorSplit }
    }
  ]

  const results = []

  for (const mode of modes) {
    const checkpointDir = `./benchmark_checkpoints/large-${mode.label.replace(/\s+/g, '-').toLowerCase()}`
    const outputDir = `./benchmark_output/large-${mode.label.replace(/\s+/g, '-').toLowerCase()}`
    cleanupDir(checkpointDir)
    cleanupDir(outputDir)

    const finetuneOptions = {
      ...finetuneOptionsBase,
      checkpointSaveDir: checkpointDir,
      outputParametersDir: outputDir
    }

    console.log(`\n${'─'.repeat(60)}`)
    console.log(`Mode: ${mode.label}`)
    console.log(`Config: ${JSON.stringify(mode.config)}`)
    console.log('─'.repeat(60))

    const client = new LlamaClient({
      files: { model: [modelPath] },
      config: mode.config,
      logger,
      opts: { stats: true }
    })

    try {
      const loadStart = Date.now()
      await client.load()
      const loadTimeMs = Date.now() - loadStart
      console.log(`Model loaded in ${loadTimeMs}ms`)

      console.log('Starting finetuning...')
      const trainStart = Date.now()
      const handle = await client.finetune(finetuneOptions)

      let lastStats = null
      handle.on('stats', stats => {
        lastStats = stats
        console.log(`  ${formatProgress(stats, finetuneOptions.numberOfEpochs)}`)
      })

      const result = await handle.await()
      const trainTimeMs = Date.now() - trainStart
      const totalTimeMs = Date.now() - loadStart

      console.log(`\nResult: ${result?.status || 'UNKNOWN'}`)
      console.log(`Load: ${loadTimeMs}ms | Train: ${trainTimeMs}ms | Total: ${totalTimeMs}ms`)

      const globalSteps = result?.stats?.global_steps || lastStats?.global_steps || 0
      const stepsPerSec = globalSteps > 0 ? (globalSteps / (trainTimeMs / 1000)) : 0

      results.push({
        label: mode.label,
        loadTimeMs,
        trainTimeMs,
        totalTimeMs,
        status: result?.status || 'UNKNOWN',
        globalSteps,
        stepsPerSec,
        trainLoss: result?.stats?.train_loss,
        valLoss: result?.stats?.val_loss
      })
    } catch (err) {
      console.error(`\n  ERROR in "${mode.label}": ${err.message}`)
      console.error('  Skipping this mode.\n')
      results.push({
        label: mode.label,
        loadTimeMs: 0,
        trainTimeMs: 0,
        totalTimeMs: 0,
        status: 'ERROR: ' + err.message,
        globalSteps: 0,
        stepsPerSec: 0
      })
    } finally {
      await client.unload().catch(() => {})
      cleanupDir(checkpointDir)
      cleanupDir(outputDir)
    }
  }

  return results
}

// ──────────────────────────────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────────────────────────────

function printBenchmarkSummary (singleGpuResult, largeModelResults) {
  console.log('\n' + '='.repeat(72))
  console.log('BENCHMARK SUMMARY')
  console.log('='.repeat(72))

  // Section 1 summary
  console.log('\n--- Section 1: Single-GPU Pause/Resume (Qwen3-32B-Q4_0) ---')
  console.log(`  Status: ${singleGpuResult.status}`)
  console.log(`  Pause phase: ${fmt(singleGpuResult.pauseTimeMs, 0)}ms`)
  console.log(`  Resume phase: ${fmt(singleGpuResult.resumeTimeMs, 0)}ms`)
  console.log(`  Total: ${fmt(singleGpuResult.totalTimeMs, 0)}ms`)
  if (singleGpuResult.stats) {
    console.log(`  Final train_loss: ${fmt(singleGpuResult.stats.train_loss, 4)}`)
    console.log(`  Final val_loss: ${fmt(singleGpuResult.stats.val_loss, 4)}`)
  }

  // Section 2 summary
  if (largeModelResults.length > 0) {
    console.log('\n--- Section 2: Large-Model Finetune (Qwen3-32B-Q4_0) ---\n')
    console.log(
      'Mode'.padEnd(28) +
      'Status'.padEnd(12) +
      'Load(ms)'.padEnd(10) +
      'Train(ms)'.padEnd(12) +
      'Total(ms)'.padEnd(12) +
      'Steps'.padEnd(8) +
      'Steps/s'.padEnd(10) +
      'Train Loss'.padEnd(12) +
      'Val Loss'
    )
    console.log('-'.repeat(104))

    for (const r of largeModelResults) {
      console.log(
        r.label.padEnd(28) +
        r.status.slice(0, 10).padEnd(12) +
        fmt(r.loadTimeMs, 0).padEnd(10) +
        fmt(r.trainTimeMs, 0).padEnd(12) +
        fmt(r.totalTimeMs, 0).padEnd(12) +
        String(r.globalSteps).padEnd(8) +
        fmt(r.stepsPerSec, 2).padEnd(10) +
        fmt(r.trainLoss, 4).padEnd(12) +
        fmt(r.valLoss, 4)
      )
    }

    const successful = largeModelResults.filter(r => r.status === 'COMPLETED' && r.trainTimeMs > 0)
    if (successful.length >= 2) {
      const baseline = successful[0]
      console.log('\nSpeedup relative to single GPU:')
      for (let i = 1; i < successful.length; i++) {
        const r = successful[i]
        const trainSpeedup = baseline.trainTimeMs / r.trainTimeMs
        const totalSpeedup = baseline.totalTimeMs / r.totalTimeMs
        const stepsSpeedup = r.stepsPerSec / baseline.stepsPerSec
        console.log(
          `  ${r.label}: ` +
          `train ${fmt(trainSpeedup, 2)}x, ` +
          `total ${fmt(totalSpeedup, 2)}x, ` +
          `steps/s ${fmt(stepsSpeedup, 2)}x`
        )
      }
    }
  }

  console.log('\n' + '='.repeat(72))
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main () {
  console.log('Multi-GPU Finetuning Benchmark (Qwen3-32B-Q4_0)')
  console.log('')
  console.log('Section 1: Single-GPU finetune with pause/resume')
  console.log('Section 2: Single vs multi-GPU finetuning speed comparison')
  console.log('')
  console.log('Usage: bare examples/finetune/multi-gpu-finetune-benchmark.js [options]')
  console.log('Options:')
  console.log('  --tensor-split=1,1   GPU split proportions (default: 1,1)')
  console.log('  --epochs=1           Epochs for large-model benchmark (default: 1)')
  console.log('  --skip-section1      Skip the single-GPU pause/resume section')
  console.log('  --skip-section2      Skip the large-model multi-GPU benchmark')
  console.log('')

  const tensorSplit = parseStringArg('tensor-split', '1,1')
  const epochs = parseIntegerArg('epochs', 1)
  const skipSection1 = hasFlag('skip-section1')
  const skipSection2 = hasFlag('skip-section2')

  const { logger, restore: restoreConsole } = createFilteredLogger()

  let singleGpuResult = null
  let largeModelResults = []

  try {
    // ── Section 1 ──
    if (!skipSection1) {
      singleGpuResult = await runSingleGpuPauseResume(logger, { epochs })
    } else {
      console.log('\n[Skipping Section 1: --skip-section1]')
      singleGpuResult = { totalTimeMs: 0, pauseTimeMs: 0, resumeTimeMs: 0, status: 'SKIPPED' }
    }

    // ── Section 2 ──
    if (!skipSection2) {
      largeModelResults = await runLargeModelFinetuneBenchmark(logger, { tensorSplit, epochs })
    } else {
      console.log('\n[Skipping Section 2: --skip-section2]')
    }

    // ── Summary ──
    printBenchmarkSummary(singleGpuResult, largeModelResults)
  } finally {
    restoreConsole()
  }
}

main().catch(error => {
  console.error('\nFatal error:', error.message)
  console.error('Stack:', error.stack)
  process.exit(1)
})

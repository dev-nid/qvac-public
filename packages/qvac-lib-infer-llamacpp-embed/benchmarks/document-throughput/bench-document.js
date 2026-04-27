'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const process = require('bare-process')
const os = require('bare-os')
const GGMLBert = require('../../index')
const { generateDocument, chunkDocument, getDocumentStats } = require('./generate-document')

const MODEL_URL = 'https://huggingface.co/ChristianAzinn/gte-large-gguf/resolve/main/gte-large_fp16.gguf'
const MODEL_FILENAME = 'gte-large_fp16.gguf'
const CHUNK_CONFIGS = [
  { label: 'small (64w, 10w overlap)', chunkSize: 64, overlap: 10 },
  { label: 'medium (128w, 20w overlap)', chunkSize: 128, overlap: 20 },
  { label: 'large (256w, 40w overlap)', chunkSize: 256, overlap: 40 }
]

const WARMUP_RUNS = 2
const BENCH_REPEATS = 3

function parseArgs (argv) {
  const parsed = {}
  for (let i = 2; i < argv.length; i++) {
    const token = argv[i]
    if (!token.startsWith('--')) continue
    const key = token.slice(2)
    const next = argv[i + 1]
    if (!next || next.startsWith('--')) {
      parsed[key] = true
    } else {
      parsed[key] = next
      i++
    }
  }
  return parsed
}

async function downloadFile (url, dest) {
  const https = require('bare-https')
  return new Promise((resolve, reject) => {
    let resolved = false
    const safeResolve = () => { if (!resolved) { resolved = true; resolve() } }
    const safeReject = (err) => { if (!resolved) { resolved = true; reject(err) } }

    const file = fs.createWriteStream(dest)
    file.on('error', (err) => {
      file.destroy()
      fs.unlink(dest, () => safeReject(err))
    })

    const req = https.request(url, (response) => {
      if ([301, 302, 307, 308].includes(response.statusCode)) {
        file.destroy()
        fs.unlink(dest, (unlinkErr) => {
          if (unlinkErr && unlinkErr.code !== 'ENOENT') return safeReject(unlinkErr)
          const redirectUrl = new URL(response.headers.location, url).href
          downloadFile(redirectUrl, dest).then(safeResolve).catch(safeReject)
        })
        return
      }
      if (response.statusCode !== 200) {
        file.destroy()
        fs.unlink(dest, () => safeReject(new Error('Download failed: ' + response.statusCode)))
        return
      }

      const total = parseInt(response.headers['content-length'], 10)
      let downloaded = 0
      response.on('data', (chunk) => {
        downloaded += chunk.length
        if (total) {
          const pct = ((downloaded / total) * 100).toFixed(1)
          const dlMB = (downloaded / 1024 / 1024).toFixed(1)
          const totMB = (total / 1024 / 1024).toFixed(1)
          process.stdout.write('\r  ' + pct + '% (' + dlMB + '/' + totMB + 'MB)')
        }
      })
      response.on('error', (err) => {
        file.destroy()
        fs.unlink(dest, () => safeReject(err))
      })
      response.pipe(file)
      file.on('close', () => {
        console.log('')
        safeResolve()
      })
    })
    req.on('error', (err) => {
      file.destroy()
      fs.unlink(dest, () => safeReject(err))
    })
    req.end()
  })
}

async function ensureModel () {
  const modelDir = path.resolve(__dirname, '../../test/model')
  const modelPath = path.join(modelDir, MODEL_FILENAME)

  if (fs.existsSync(modelPath)) {
    const stats = fs.statSync(modelPath)
    console.log('Model found: ' + MODEL_FILENAME + ' (' + (stats.size / 1024 / 1024).toFixed(1) + 'MB)')
    return modelPath
  }

  fs.mkdirSync(modelDir, { recursive: true })
  console.log('Downloading model: ' + MODEL_FILENAME + '...')
  await downloadFile(MODEL_URL, modelPath)
  console.log('Download complete.')
  return modelPath
}

function silentLogger () {
  return { error () {}, warn () {}, info () {}, debug () {} }
}

async function loadModel (modelPath, device, batchSize, splitMode, tensorSplit) {
  const isDarwinX64 = os.platform() === 'darwin' && os.arch() === 'x64'
  if (isDarwinX64) device = 'cpu'

  const config = {
    device,
    gpu_layers: device === 'cpu' ? '0' : '999',
    batch_size: String(batchSize)
  }

  if (splitMode && splitMode !== 'none') {
    config['split-mode'] = splitMode
  }
  if (tensorSplit) {
    config['tensor-split'] = tensorSplit
  }

  if (os.platform() === 'android') {
    config.flash_attn = 'off'
  }

  const model = new GGMLBert({
    files: { model: [modelPath] },
    config,
    logger: silentLogger(),
    opts: { stats: true }
  })

  const loadStart = process.hrtime()
  await model.load()
  const loadElapsed = hrtimeMs(loadStart)

  return { model, loadMs: loadElapsed, device }
}

function hrtimeMs (start) {
  const diff = process.hrtime(start)
  return diff[0] * 1000 + diff[1] / 1e6
}

async function embedChunks (model, chunks) {
  const response = await model.run(chunks)
  const rawEmbeddings = await response._finishPromise
  const stats = response.stats || {}
  return { rawEmbeddings, stats }
}

async function benchmarkChunkConfig (model, chunks, configLabel) {
  console.log('\n  Chunk config: ' + configLabel)
  console.log('  Chunks to embed: ' + chunks.length)

  for (let i = 0; i < WARMUP_RUNS; i++) {
    await embedChunks(model, chunks)
  }

  const timings = []
  const tpsValues = []
  let lastEmbeddings = null

  for (let i = 0; i < BENCH_REPEATS; i++) {
    const runStart = process.hrtime()
    const { rawEmbeddings, stats } = await embedChunks(model, chunks)
    const runMs = hrtimeMs(runStart)
    timings.push(runMs)
    lastEmbeddings = rawEmbeddings
    if (stats.tokens_per_second) tpsValues.push(stats.tokens_per_second)
  }

  const embeddingCount = lastEmbeddings[0] ? lastEmbeddings[0].length : 0
  const embeddingDim = embeddingCount > 0 ? lastEmbeddings[0][0].length : 0

  const avgMs = timings.reduce((a, b) => a + b, 0) / timings.length
  const minMs = Math.min(...timings)
  const maxMs = Math.max(...timings)
  const avgTps = tpsValues.length > 0
    ? tpsValues.reduce((a, b) => a + b, 0) / tpsValues.length
    : null

  const chunksPerSec = (chunks.length / (avgMs / 1000))

  return {
    configLabel,
    numChunks: chunks.length,
    embeddingCount,
    embeddingDim,
    avgMs: round(avgMs),
    minMs: round(minMs),
    maxMs: round(maxMs),
    avgTps: avgTps ? round(avgTps) : null,
    chunksPerSec: round(chunksPerSec),
    timings: timings.map(round)
  }
}

function round (val, decimals) {
  if (decimals == null) decimals = 2
  const factor = Math.pow(10, decimals)
  return Math.round(val * factor) / factor
}

function printResults (docStats, results, loadMs, device) {
  console.log('\n' + '='.repeat(72))
  console.log('DOCUMENT THROUGHPUT BENCHMARK RESULTS')
  console.log('='.repeat(72))

  console.log('\nDocument:')
  console.log('  Words:      ' + docStats.words)
  console.log('  Characters: ' + docStats.chars)
  console.log('  Paragraphs: ' + docStats.paragraphs)
  console.log('  Est. pages: ' + docStats.pages)

  console.log('\nModel: ' + MODEL_FILENAME)
  console.log('Device: ' + device)
  console.log('Load time: ' + round(loadMs) + 'ms')
  console.log('Warmup runs: ' + WARMUP_RUNS)
  console.log('Bench repeats: ' + BENCH_REPEATS)

  console.log('\n' + '-'.repeat(72))
  console.log(padRight('Chunk Config', 30) + padRight('Chunks', 8) + padRight('Avg(ms)', 10) + padRight('Min(ms)', 10) + padRight('Max(ms)', 10) + padRight('TPS', 10) + 'Chunks/s')
  console.log('-'.repeat(72))

  for (const r of results) {
    console.log(
      padRight(r.configLabel, 30) +
      padRight(String(r.numChunks), 8) +
      padRight(String(r.avgMs), 10) +
      padRight(String(r.minMs), 10) +
      padRight(String(r.maxMs), 10) +
      padRight(r.avgTps != null ? String(r.avgTps) : 'n/a', 10) +
      String(r.chunksPerSec)
    )
  }

  console.log('-'.repeat(72))

  for (const r of results) {
    console.log('\n  [' + r.configLabel + ']')
    console.log('    Embeddings produced: ' + r.embeddingCount + ' x ' + r.embeddingDim + 'd')
    console.log('    Individual run times: ' + r.timings.join('ms, ') + 'ms')
    const docsPerMin = round((60000 / r.avgMs), 1)
    console.log('    Estimated 10-page docs/min: ' + docsPerMin)
  }

  console.log('\n' + '='.repeat(72))
}

function padRight (str, len) {
  if (str.length >= len) return str
  return str + ' '.repeat(len - str.length)
}

function writeReport (docStats, results, loadMs, device, splitMode, tensorSplit) {
  const reportDir = path.resolve(__dirname, 'results')
  fs.mkdirSync(reportDir, { recursive: true })

  const now = new Date()
  const dateStr = now.toISOString().slice(0, 10)
  const reportPath = path.join(reportDir, 'document-throughput-' + dateStr + '.json')

  const report = {
    timestamp: now.toISOString(),
    model: MODEL_FILENAME,
    device,
    splitMode,
    tensorSplit: tensorSplit || null,
    loadMs: round(loadMs),
    warmupRuns: WARMUP_RUNS,
    benchRepeats: BENCH_REPEATS,
    document: docStats,
    platform: {
      os: os.platform(),
      arch: os.arch()
    },
    results
  }

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
  console.log('\nReport saved to: ' + reportPath)
  return reportPath
}

async function main () {
  const args = parseArgs(process.argv)
  const device = args.device || 'gpu'
  const batchSize = parseInt(args['batch-size'] || '2048', 10)
  const splitMode = args['split-mode'] || 'none'
  const tensorSplit = args['tensor-split'] || null
  const multiplier = parseInt(args.multiply || '1', 10)
  if (multiplier < 1 || !Number.isFinite(multiplier)) {
    throw new Error('--multiply must be a positive integer (got: ' + args.multiply + ')')
  }

  console.log('=== 10-Page Document Throughput Benchmark ===\n')

  const baseDocument = generateDocument()
  const document = multiplier > 1
    ? Array(multiplier).fill(baseDocument).join('\n\n')
    : baseDocument
  const docStats = getDocumentStats(document)
  const multiplierLabel = multiplier > 1 ? ' (' + multiplier + 'x)' : ''
  console.log('Generated document: ' + docStats.words + ' words, ~' + docStats.pages + ' pages' + multiplierLabel)

  const modelPath = await ensureModel()

  const splitLabel = splitMode !== 'none' ? ', split-mode=' + splitMode : ''
  const tensorLabel = tensorSplit ? ', tensor-split=' + tensorSplit : ''
  console.log('\nLoading model (device=' + device + ', batch_size=' + batchSize + splitLabel + tensorLabel + ')...')
  const { model, loadMs, device: resolvedDevice } = await loadModel(modelPath, device, batchSize, splitMode, tensorSplit)
  console.log('Model loaded in ' + round(loadMs) + 'ms')

  const results = []

  try {
    for (const cfg of CHUNK_CONFIGS) {
      const chunks = chunkDocument(document, cfg.chunkSize, cfg.overlap)
      const result = await benchmarkChunkConfig(model, chunks, cfg.label)
      results.push(result)
    }

    printResults(docStats, results, loadMs, resolvedDevice)
    writeReport(docStats, results, loadMs, resolvedDevice, splitMode, tensorSplit)
  } finally {
    console.log('\nUnloading model...')
    await model.unload()
    console.log('Done.')
  }
}

main().catch((error) => {
  console.error('Benchmark failed:')
  console.error(error.stack || String(error))
  process.exit(1)
})

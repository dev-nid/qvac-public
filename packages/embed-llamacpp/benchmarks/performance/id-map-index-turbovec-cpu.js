'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const process = require('bare-process')
const IdMapIndex = require('../../idMapIndex')
const { elapsedMs, round } = require('./math')

const DIM = 384
const VECTOR_COUNT = 10000
const QUERY_COUNT = 128
const K = 10
const BIT_WIDTHS = [4, 8, 32]
const FILTER_COUNT = 1000
const IVF_LISTS = 100
const IVF_NPROBE = 10
const DELTA_COUNT = 128
const MEASURED_SEARCH_RUNS = 3
const REPORT_PATH = path.join(__dirname, 'id-map-index-turbovec-cpu-report.md')
const TMP_DIR = path.join(__dirname, '.tmp-id-map-index-turbovec')
const MAIN_BASELINE_COMMIT = '168e2dd15'
const MAIN_BUILD_STATUS = 'built successfully with npm install, bare-make generate, bare-make build, and bare-make install'

function ensureTmpDir () {
  if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR)
}

function cleanupFile (file) {
  if (fs.existsSync(file)) fs.unlinkSync(file)
}

function createRng (seed) {
  let state = seed >>> 0
  return function next () {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

function createNormalizedVectors (n, dim, seed) {
  const rng = createRng(seed)
  const vectors = new Float32Array(n * dim)
  for (let row = 0; row < n; row++) {
    const offset = row * dim
    let normSq = 0
    for (let col = 0; col < dim; col++) {
      const value = rng() * 2 - 1
      vectors[offset + col] = value
      normSq += value * value
    }
    const invNorm = 1 / Math.sqrt(normSq)
    for (let col = 0; col < dim; col++) {
      vectors[offset + col] *= invNorm
    }
  }
  return vectors
}

function createIds (n, base) {
  const ids = new BigUint64Array(n)
  const start = BigInt(base)
  for (let i = 0; i < n; i++) ids[i] = start + BigInt(i)
  return ids
}

function createQueries (vectors, nVectors, dim, nQueries) {
  const queries = new Float32Array(nQueries * dim)
  for (let i = 0; i < nQueries; i++) {
    const source = (i * 73) % nVectors
    queries.set(vectors.subarray(source * dim, source * dim + dim), i * dim)
  }
  return queries
}

function createFilterIds (ids, n) {
  const allowed = new BigUint64Array(n)
  for (let i = 0; i < n; i++) allowed[i] = ids[i]
  return allowed
}

function measureSync (fn) {
  const start = process.hrtime()
  const value = fn()
  return { ms: elapsedMs(start), value }
}

async function measureAsync (fn) {
  const start = process.hrtime()
  const value = await fn()
  return { ms: elapsedMs(start), value }
}

function median (values) {
  const sorted = values.slice().sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

function measureMedianMs (fn) {
  const times = []
  for (let i = 0; i < MEASURED_SEARCH_RUNS; i++) {
    times.push(measureSync(fn).ms)
  }
  return median(times)
}

function qps (nQueries, ms) {
  return (nQueries / ms) * 1000
}

function mb (bytes) {
  return bytes / (1024 * 1024)
}

function kb (bytes) {
  return bytes / 1024
}

function fileSize (file) {
  return fs.statSync(file).size
}

function recallAtK (exactIds, approxIds, nQueries, k) {
  let hits = 0
  let total = 0
  for (let row = 0; row < nQueries; row++) {
    const expected = new Set()
    for (let col = 0; col < k; col++) {
      expected.add(exactIds[row * k + col].toString())
    }
    for (let col = 0; col < k; col++) {
      if (expected.has(approxIds[row * k + col].toString())) hits++
      total++
    }
  }
  return total === 0 ? 0 : hits / total
}

function bruteForceSearch (vectors, ids, queries, nVectors, dim, nQueries, k) {
  const outIds = new BigUint64Array(nQueries * k)
  const outScores = new Float32Array(nQueries * k)
  for (let row = 0; row < nQueries; row++) {
    const bestScores = new Float32Array(k)
    const bestIds = new BigUint64Array(k)
    for (let slot = 0; slot < k; slot++) {
      bestScores[slot] = -Infinity
      bestIds[slot] = (1n << 64n) - 1n
    }

    const queryOffset = row * dim
    for (let vectorIndex = 0; vectorIndex < nVectors; vectorIndex++) {
      const vectorOffset = vectorIndex * dim
      let score = 0
      for (let col = 0; col < dim; col++) {
        score += queries[queryOffset + col] * vectors[vectorOffset + col]
      }
      for (let slot = 0; slot < k; slot++) {
        if (score <= bestScores[slot]) continue
        for (let shift = k - 1; shift > slot; shift--) {
          bestScores[shift] = bestScores[shift - 1]
          bestIds[shift] = bestIds[shift - 1]
        }
        bestScores[slot] = score
        bestIds[slot] = ids[vectorIndex]
        break
      }
    }

    outScores.set(bestScores, row * k)
    outIds.set(bestIds, row * k)
  }
  return { ids: outIds, scores: outScores }
}

function toMarkdown (report) {
  const lines = []
  lines.push('# IdMapIndex TurboVec CPU Benchmark')
  lines.push('')
  lines.push(`- Generated: ${report.generatedAt}`)
  lines.push(`- Command: \`bare benchmarks/performance/id-map-index-turbovec-cpu.js\``)
  lines.push(`- Runtime: ${report.runtime}`)
  lines.push(`- Platform: ${report.platform}`)
  lines.push(`- Dataset: ${report.vectorCount} vectors x ${report.dim} dimensions`)
  lines.push(`- Queries: ${report.queryCount}, top-k: ${report.k}`)
  lines.push(`- IVF: ${report.ivfLists} lists, nProbe ${report.ivfNprobe}`)
  lines.push(`- Search timings: median of ${report.searchRuns} measured runs after one warmup`)
  lines.push('')
  lines.push('## Comparison To `main`')
  lines.push('')
  lines.push(
    `Built and checked \`main\` at \`${report.mainBaselineCommit}\` ` +
      `(${report.mainBuildStatus}): ` +
      '`packages/embed-llamacpp` does not contain the `IdMapIndex` JS API, ' +
      'the vector-index native binding, or matching tests. There is no same-API ' +
      'CPU vector-index baseline to run on `main`.'
  )
  lines.push('')
  lines.push(
    'No local GGUF model was present in either worktree, so embedding-throughput ' +
      'comparison was not run. Because `main` only exposes embeddings, the closest ' +
      'same-data retrieval baseline is JS brute-force dot-product search over the ' +
      'embedding arrays.'
  )
  lines.push('')
  lines.push(`JS brute-force exact search baseline: ${report.bruteForceQps} q/s (${report.bruteForceMsPerQuery} ms/query).`)
  lines.push('')
  lines.push('## Main CPU Results')
  lines.push('')
  lines.push('| Storage | .tvim MB | Ingest vectors/s | Exact q/s | Exact ms/query | IVF build ms | IVF q/s | IVF ms/query | IVF recall@10 |')
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
  for (const item of report.results) {
    lines.push(
      `| ${item.storage} | ${item.fileMb} | ${item.ingestVectorsPerSec}` +
      ` | ${item.exactQps} | ${item.exactMsPerQuery}` +
      ` | ${item.ivfBuildMs} | ${item.ivfQps} | ${item.ivfMsPerQuery}` +
      ` | ${item.ivfRecallAt10} |`
    )
  }
  lines.push('')
  lines.push('## Persistence, Filter, Mmap, Delta')
  lines.push('')
  lines.push('| Storage | Load ms | Mmap load ms | Mmap exact q/s | Prepared filter build ms | Prepared filter q/s | addLogged 128 ms | removeLogged 128 total ms | Delta KB |')
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
  for (const item of report.results) {
    lines.push(
      `| ${item.storage} | ${item.loadMs} | ${item.mmapLoadMs}` +
      ` | ${item.mmapExactQps} | ${item.filterBuildMs}` +
      ` | ${item.preparedFilterQps} | ${item.addLoggedMs}` +
      ` | ${item.removeLoggedMs} | ${item.deltaKb} |`
    )
  }
  lines.push('')
  lines.push('## Notes')
  lines.push('')
  for (const note of report.notes) lines.push(`- ${note}`)
  lines.push('')
  return `${lines.join('\n')}\n`
}

function buildNotes (results, bruteForceQps) {
  const notes = []
  const f32 = results.find((item) => item.storage === 'f32')
  const fastestExact = results.reduce((best, item) => {
    return best === null || item.exactQps > best.exactQps ? item : best
  }, null)
  if (fastestExact !== null) {
    notes.push(`${fastestExact.storage} had the fastest exact search in this run at ${fastestExact.exactQps} q/s.`)
  }
  if (fastestExact !== null && f32 && fastestExact.storage !== 'f32') {
    notes.push(
      `${fastestExact.storage} exact search was ${round(fastestExact.exactQps / f32.exactQps, 2)}x faster than current f32.`
    )
  }
  if (fastestExact !== null && bruteForceQps > 0) {
    notes.push(
      `${fastestExact.storage} exact search was ${round(fastestExact.exactQps / bruteForceQps, 2)}x faster than JS brute-force search.`
    )
  }
  for (const item of results) {
    if (f32 && item.storage !== 'f32') {
      notes.push(
        `${item.storage} snapshot size is ${round(item.fileMb / f32.fileMb, 2)}x of f32 for this dataset.`
      )
    }
    notes.push(
      `${item.storage} IVF search is ${round(item.exactMsPerQuery / item.ivfMsPerQuery, 2)}x faster than exact search at recall ${item.ivfRecallAt10}.`
    )
  }
  notes.push('Lower bit widths reduce persisted size; recall and latency depend on vector distribution and IVF settings.')
  notes.push('removeLogged is measured as 128 individual durable remove appends, not a bulk delete API.')
  notes.push('This benchmark uses normalized synthetic vectors, so it measures index mechanics rather than model embedding quality.')
  return notes
}

async function runCase (bitWidth, vectors, ids, queries, filterIds, deltaVectors) {
  const storage = bitWidth === 32 ? 'f32' : `q${bitWidth}`
  const snapshot = path.join(TMP_DIR, `id-map-index-${storage}.tvim`)
  const delta = path.join(TMP_DIR, `id-map-index-${storage}.tvid`)
  cleanupFile(snapshot)
  cleanupFile(delta)

  let idx = null
  let loaded = null
  let mmap = null
  let filter = null
  try {
    idx = new IdMapIndex({ dim: DIM, bitWidth })

    const add = measureSync(() => {
      idx.addWithIds(vectors, ids)
    })
    const ingestVectorsPerSec = qps(VECTOR_COUNT, add.ms)

    idx.search(queries.subarray(0, DIM * 8), K)
    const exactResult = idx.search(queries, K)
    const exactMs = measureMedianMs(() => {
      idx.search(queries, K)
    })

    const filterBuild = measureSync(() => {
      filter = idx.prepareFilter(filterIds)
    })
    filter.search(queries.subarray(0, DIM * 8), K)
    const preparedFilterMs = measureMedianMs(() => {
      filter.search(queries, K)
    })

    const ivfBuild = measureSync(() => {
      idx.buildIvf(IVF_LISTS, 1)
    })
    idx.searchIvf(queries.subarray(0, DIM * 8), K, IVF_NPROBE)
    const ivfResult = idx.searchIvf(queries, K, IVF_NPROBE)
    const ivfMs = measureMedianMs(() => {
      idx.searchIvf(queries, K, IVF_NPROBE)
    })

    measureSync(() => {
      idx.write(snapshot)
    })
    const sizeBytes = fileSize(snapshot)

    const load = await measureAsync(() => IdMapIndex.load(snapshot))
    loaded = load.value
    const mmapLoad = await measureAsync(() => IdMapIndex.loadMmap(snapshot))
    mmap = mmapLoad.value
    mmap.search(queries.subarray(0, DIM * 8), K)
    const mmapExactMs = measureMedianMs(() => {
      mmap.search(queries, K)
    })

    const deltaIds = createIds(DELTA_COUNT, 9000000 + bitWidth * 1000)
    const addLogged = measureSync(() => {
      idx.addLogged(deltaVectors, deltaIds, delta)
    })
    const removeLogged = measureSync(() => {
      for (let i = 0; i < DELTA_COUNT; i++) {
        idx.removeLogged(ids[i], delta)
      }
    })

    return {
      storage,
      fileMb: round(mb(sizeBytes), 2),
      ingestVectorsPerSec: Math.round(ingestVectorsPerSec),
      exactQps: Math.round(qps(QUERY_COUNT, exactMs)),
      exactMsPerQuery: round(exactMs / QUERY_COUNT, 3),
      ivfBuildMs: round(ivfBuild.ms, 2),
      ivfQps: Math.round(qps(QUERY_COUNT, ivfMs)),
      ivfMsPerQuery: round(ivfMs / QUERY_COUNT, 3),
      ivfRecallAt10: round(recallAtK(exactResult.ids, ivfResult.ids, QUERY_COUNT, K), 3),
      loadMs: round(load.ms, 2),
      mmapLoadMs: round(mmapLoad.ms, 2),
      mmapExactQps: Math.round(qps(QUERY_COUNT, mmapExactMs)),
      filterBuildMs: round(filterBuild.ms, 2),
      preparedFilterQps: Math.round(qps(QUERY_COUNT, preparedFilterMs)),
      addLoggedMs: round(addLogged.ms, 2),
      removeLoggedMs: round(removeLogged.ms, 2),
      deltaKb: round(kb(fileSize(delta)), 1)
    }
  } finally {
    if (filter !== null) await filter.dispose()
    if (loaded !== null) await loaded.dispose()
    if (mmap !== null) await mmap.dispose()
    if (idx !== null) await idx.dispose()
    cleanupFile(snapshot)
    cleanupFile(delta)
  }
}

async function main () {
  ensureTmpDir()
  console.log('Generating synthetic vectors...')
  const vectors = createNormalizedVectors(VECTOR_COUNT, DIM, 0x5eed1234)
  const ids = createIds(VECTOR_COUNT, 1000000)
  const queries = createQueries(vectors, VECTOR_COUNT, DIM, QUERY_COUNT)
  const filterIds = createFilterIds(ids, FILTER_COUNT)
  const deltaVectors = createNormalizedVectors(DELTA_COUNT, DIM, 0x9e3779b9)

  console.log('Benchmarking JS brute-force baseline...')
  bruteForceSearch(vectors, ids, queries.subarray(0, DIM * 8), VECTOR_COUNT, DIM, 8, K)
  const bruteForce = measureSync(() => {
    bruteForceSearch(vectors, ids, queries, VECTOR_COUNT, DIM, QUERY_COUNT, K)
  })

  const results = []
  for (const bitWidth of BIT_WIDTHS) {
    console.log(`Benchmarking bitWidth=${bitWidth}...`)
    results.push(await runCase(bitWidth, vectors, ids, queries, filterIds, deltaVectors))
  }

  const report = {
    generatedAt: new Date().toISOString(),
    runtime: process.version || 'unknown',
    platform: `${process.platform || 'unknown'} ${process.arch || 'unknown'}`,
    mainBaselineCommit: MAIN_BASELINE_COMMIT,
    mainBuildStatus: MAIN_BUILD_STATUS,
    bruteForceQps: Math.round(qps(QUERY_COUNT, bruteForce.ms)),
    bruteForceMsPerQuery: round(bruteForce.ms / QUERY_COUNT, 3),
    dim: DIM,
    vectorCount: VECTOR_COUNT,
    queryCount: QUERY_COUNT,
    k: K,
    ivfLists: IVF_LISTS,
    ivfNprobe: IVF_NPROBE,
    searchRuns: MEASURED_SEARCH_RUNS,
    results,
    notes: buildNotes(results, qps(QUERY_COUNT, bruteForce.ms))
  }
  fs.writeFileSync(REPORT_PATH, toMarkdown(report))
  console.log(`Wrote ${REPORT_PATH}`)
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err)
  process.exit(1)
})

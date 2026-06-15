'use strict'
//
// turbovec-vs-hyperdb: side-by-side perf on identical synthetic vectors.
//
// Three adapters run end-to-end against the SAME generated vectors so
// numbers are directly comparable on this host:
//
//   1. TurboVecHybridAdapter            (POC, naive scalar full-f32)
//   2. HyperDBAdapter — production defaults (NUM_CENTROIDS=16, BUCKET_SIZE=50)
//      → at N > 800 this silently evicts via FIFO; recall craters but
//        ingest/query is "fast". Included to show what shipping users see.
//   3. HyperDBAdapter — retain-full (BUCKET_SIZE bumped so cap >= N)
//      → matches the REPORT.md "tuned to retain everything" config that
//        hits recall ~0.986 but at multi-second p50 on real corpora.
//
// Recall is NOT measured here — synthetic random vectors have no semantic
// ground truth. The architectural numbers (ingest / query latency / RAM /
// file size) are the comparison; recall numbers live in REPORT.md.
//
// Usage:
//   bare examples/turbovec-vs-hyperdb.js                 # N=3633, dim=1024, 50 queries
//   bare examples/turbovec-vs-hyperdb.js --n=10000       # bigger N
//   bare examples/turbovec-vs-hyperdb.js --skip-hyperdb-full --n=50000
//                                                        # skip the slowest run
//   bare examples/turbovec-vs-hyperdb.js --only=turbo    # just one adapter

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const Corestore = require('corestore')

const TurboVecHybridAdapter =
  require('../src/adapters/database/TurboVecHybridAdapter')
const HyperDBAdapter =
  require('../src/adapters/database/HyperDBAdapter')

// ----- args ---------------------------------------------------------------

function parseArgs (argv) {
  const out = {
    n: 3633, dim: 1024, queries: 50, k: 10, seed: 0xdeadbeef,
    'skip-hyperdb-default': false,
    'skip-hyperdb-full': false,
    only: null
  }
  for (const a of argv.slice(2)) {
    if (a.startsWith('--no-') || a.startsWith('--skip-')) {
      out[a.replace(/^--/, '')] = true
      continue
    }
    const m = /^--(\S+?)=(.+)$/.exec(a)
    if (m) out[m[1]] = /^\d+$/.test(m[2]) ? Number(m[2]) : m[2]
  }
  return out
}

const args = parseArgs(Bare.argv)
const { n: N, dim: DIM, queries: NQ, k: K, seed: SEED } = args
const MODEL_ID = 'synthetic-bench'

// ----- shared utils -------------------------------------------------------

function makePrng (seed) {
  let s = seed >>> 0
  return () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) / 0xffffffff)
  }
}

function l2normalize (v) {
  let acc = 0
  for (let i = 0; i < v.length; i++) acc += v[i] * v[i]
  const n = Math.sqrt(acc) || 1
  for (let i = 0; i < v.length; i++) v[i] /= n
}

function buildVecs (n, dim, rng) {
  const buf = new Float32Array(n * dim)
  for (let i = 0; i < n; i++) {
    const slice = buf.subarray(i * dim, (i + 1) * dim)
    for (let j = 0; j < dim; j++) slice[j] = rng() * 2 - 1
    l2normalize(slice)
  }
  return buf
}

function buildDocs (vectors, n, dim) {
  const docs = new Array(n)
  for (let i = 0; i < n; i++) {
    docs[i] = {
      id: `v${i}`,
      // Unique content + contentHash per doc. HyperDBAdapter dedupes by
      // contentHash internally; empty/duplicated values would collapse
      // the whole batch to a single survivor.
      content: `doc ${i}`,
      contentHash: `h-${i}`,
      embeddingModelId: MODEL_ID,
      embedding: Array.from(vectors.subarray(i * dim, (i + 1) * dim))
    }
  }
  return docs
}

function rssMB () { return os.memoryUsage().rss / 1024 / 1024 }
function percentile (sorted, p) {
  if (sorted.length === 0) return NaN
  const idx = Math.min(sorted.length - 1,
    Math.floor(sorted.length * (p / 100)))
  return sorted[idx]
}

// ----- runner -------------------------------------------------------------

// Generic adapter perf harness. Caller passes a setup() returning an
// open BaseDBAdapter + a teardown() that closes the underlying store.
async function runAdapter (label, vectors, queryVecs, setup) {
  console.log(`\n--- ${label} ---`)
  const before = rssMB()

  const t0open = Date.now()
  const { adapter, teardown } = await setup()
  const tOpen = Date.now() - t0open
  console.log(`  open                   ${tOpen.toString().padStart(7)} ms   rss=${rssMB().toFixed(1)} MB`)

  const docs = buildDocs(vectors, N, DIM)
  const t0ing = Date.now()
  const saveResults = await adapter.saveEmbeddings(docs)
  const tIngest = Date.now() - t0ing
  const fulfilled = Array.isArray(saveResults)
    ? saveResults.filter((r) => r && r.status === 'fulfilled').length
    : -1
  const dupes = Array.isArray(saveResults)
    ? saveResults.filter((r) => r && (r.status === 'duplicate' || r.duplicate)).length
    : 0
  const ingestRSS = rssMB()
  console.log(`  ingest (N=${N})           ${tIngest.toString().padStart(7)} ms   rss=${ingestRSS.toFixed(1)} MB   throughput=${Math.round(N / (tIngest / 1000))} vecs/s`)
  console.log(`  saveEmbeddings return:  fulfilled=${fulfilled}  duplicates=${dupes}  total=${Array.isArray(saveResults) ? saveResults.length : 'non-array'}`)

  // Sanity: count docs actually persisted. For HyperDBAdapter with a
  // defaults config, this catches silent FIFO evictions; for both
  // adapters, it catches "async-buffered, not actually flushed" cases.
  let actuallyPersisted = '?'
  try {
    if (adapter.idx && typeof adapter.idx.length === 'number') {
      actuallyPersisted = adapter.idx.length
    } else if (adapter.db && adapter.documentsTable) {
      const snap = adapter.db.snapshot()
      const rows = await snap.find(adapter.documentsTable).toArray()
      actuallyPersisted = rows.length
    }
  } catch (_) {}
  console.log(`  -> docs actually queryable: ${actuallyPersisted} / ${N}`)

  // Warmup.
  const warmup = Math.min(5, NQ)
  for (let i = 0; i < warmup; i++) {
    const q = Array.from(queryVecs.subarray(i * DIM, (i + 1) * DIM))
    await adapter.search('', q, { topK: K })
  }

  // Per-query timings (Date.now resolution noise) — used for tail stats.
  const latencies = []
  for (let i = 0; i < NQ; i++) {
    const q = Array.from(queryVecs.subarray(i * DIM, (i + 1) * DIM))
    const s = Date.now()
    await adapter.search('', q, { topK: K })
    latencies.push(Date.now() - s)
  }
  latencies.sort((a, b) => a - b)
  const p50 = percentile(latencies, 50)
  const p95 = percentile(latencies, 95)
  const p99 = percentile(latencies, 99)

  // Bulk for sub-ms mean (HyperDB at retain-full will be many ms per query
  // so bulk mostly matters for fast adapters).
  const bulkReps = Math.max(1, Math.ceil(200 / NQ))
  const bulkTotal = NQ * bulkReps
  const t0bulk = Date.now()
  for (let r = 0; r < bulkReps; r++) {
    for (let i = 0; i < NQ; i++) {
      const q = Array.from(queryVecs.subarray(i * DIM, (i + 1) * DIM))
      await adapter.search('', q, { topK: K })
    }
  }
  const tBulk = Date.now() - t0bulk
  const meanQ = tBulk / bulkTotal

  console.log(`  query mean (bulk x${bulkTotal})  ${meanQ.toFixed(3).padStart(7)} ms`)
  console.log(`  query p50 / p95 / p99   ${p50.toString().padStart(3)} / ${p95.toString().padStart(3)} / ${p99.toString().padStart(3)} ms`)
  console.log(`  rss delta over baseline ${(ingestRSS - before).toFixed(1)} MB`)

  await teardown()

  return { label, tOpen, tIngest, ingestRSS, meanQ, p50, p95, p99 }
}

// ----- adapter factories --------------------------------------------------

function freshTmpRoot (tag) {
  const dir = path.join(os.tmpdir(),
    `bench-${tag}-${(os.pid && os.pid()) || 0}-${Date.now()}`)
  fs.mkdirSync(dir, { recursive: true })
  return dir
}

function turbovecFactory () {
  const tmpRoot = freshTmpRoot('turbo')
  const indexPath = path.join(tmpRoot, 'idx.tvim')
  const storeDir = path.join(tmpRoot, 'corestore')
  const store = new Corestore(storeDir)
  const adapter = new TurboVecHybridAdapter({
    store, dim: DIM, bitWidth: 4, indexPath, embeddingModelId: MODEL_ID
  })
  return {
    setup: async () => {
      await adapter.ready()
      return {
        adapter,
        teardown: async () => {
          await adapter.close()
          await store.close()
          fs.rmSync(tmpRoot, { recursive: true, force: true })
        }
      }
    }
  }
}

function hyperdbFactory (config) {
  const tmpRoot = freshTmpRoot(config.tag)
  const storeDir = path.join(tmpRoot, 'corestore')
  const store = new Corestore(storeDir)
  const adapter = new HyperDBAdapter({
    store,
    dbName: 'bench',
    NUM_CENTROIDS: config.NUM_CENTROIDS,
    BUCKET_SIZE: config.BUCKET_SIZE
  })
  return {
    setup: async () => {
      await adapter.ready()
      return {
        adapter,
        teardown: async () => {
          await adapter.close()
          await store.close()
          fs.rmSync(tmpRoot, { recursive: true, force: true })
        }
      }
    }
  }
}

// ----- main ---------------------------------------------------------------

async function main () {
  console.log('=== turbovec-vs-hyperdb (synthetic corpus) ===')
  console.log(`N=${N} vectors, dim=${DIM}, queries=${NQ}, k=${K}, seed=0x${SEED.toString(16)}`)
  console.log(`baseline rss=${rssMB().toFixed(1)} MB`)

  const rng = makePrng(SEED)
  const vectors  = buildVecs(N, DIM, rng)
  const queries  = buildVecs(NQ, DIM, rng)

  const results = []

  if (args.only == null || args.only === 'turbo') {
    results.push(await runAdapter(
      'TurboVecHybridAdapter (POC, scalar f32)',
      vectors, queries,
      turbovecFactory().setup))
  }

  if ((args.only == null || args.only === 'hyperdb-default') &&
      !args['skip-hyperdb-default']) {
    results.push(await runAdapter(
      `HyperDBAdapter (defaults: 16c x 50)   -- cap=800, evicts ${Math.max(0, N - 800)} docs`,
      vectors, queries,
      hyperdbFactory({
        tag: 'hyperdb-default', NUM_CENTROIDS: 16, BUCKET_SIZE: 50
      }).setup))
  }

  if ((args.only == null || args.only === 'hyperdb-full') &&
      !args['skip-hyperdb-full']) {
    const bucketSize = Math.ceil(N / 16) + 10
    results.push(await runAdapter(
      `HyperDBAdapter (retain-full: 16c x ${bucketSize})`,
      vectors, queries,
      hyperdbFactory({
        tag: 'hyperdb-full', NUM_CENTROIDS: 16, BUCKET_SIZE: bucketSize
      }).setup))
  }

  // ----- summary table -----
  console.log('\n=== summary ===')
  console.log(
    'adapter'.padEnd(56) +
    'open(ms)'.padStart(10) +
    'ingest(ms)'.padStart(12) +
    'vecs/s'.padStart(10) +
    'rss(MB)'.padStart(10) +
    'q-mean(ms)'.padStart(12) +
    'p50'.padStart(6) +
    'p95'.padStart(6) +
    'p99'.padStart(6))
  console.log('-'.repeat(56 + 10 + 12 + 10 + 10 + 12 + 6 + 6 + 6))
  for (const r of results) {
    console.log(
      r.label.slice(0, 56).padEnd(56) +
      r.tOpen.toString().padStart(10) +
      r.tIngest.toString().padStart(12) +
      Math.round(N / (r.tIngest / 1000)).toString().padStart(10) +
      r.ingestRSS.toFixed(0).toString().padStart(10) +
      r.meanQ.toFixed(3).padStart(12) +
      r.p50.toString().padStart(6) +
      r.p95.toString().padStart(6) +
      r.p99.toString().padStart(6))
  }
  console.log()
  console.log('note: recall NOT measured; synthetic random vectors have no ground truth.')
  console.log('      see REPORT.md for recall numbers from real corpora.')
}

main().catch((err) => {
  console.error(err)
  Bare.exit(1)
})

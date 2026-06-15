'use strict'
//
// turbovec-perf: synthetic-data performance probe for the POC.
//
// What this measures:
//   - Ingest throughput (vectors/sec) via TurboVecHybridAdapter.saveEmbeddings
//   - Persist (write) latency to a .tvim file
//   - Open + load latency
//   - Query p50 / p95 / p99 / mean over `--queries` random L2-normalized
//     query vectors, top-k = 10
//   - Peak RSS sampled after each phase
//
// What this DOES NOT measure (intentionally):
//   - Recall. No ground truth without a real corpus; the bench harness
//     under benchmarks/turbovec-eval/ (not on this machine) is where the
//     real recall@10 vs gold lives. The numbers from this script tell
//     you "does the architecture handle N x D" and the order-of-magnitude
//     cost of the POC's naive scalar full-f32 path, nothing more.
//
// POC scope reminder: the index is full f32 + scalar dot product. These
// numbers are the BASELINE the optimization phase (quantization, SIMD,
// Vulkan/Metal) measures improvement against. Per prompt.md they're
// "expected to be slow".
//
// Usage:
//   bare examples/turbovec-perf.js                # default: 3633 vecs, 1024 dim, 100 queries
//   bare examples/turbovec-perf.js --n=10000      # bump corpus size
//   bare examples/turbovec-perf.js --dim=384      # smaller dim
//   bare examples/turbovec-perf.js --queries=500  # more query samples

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const Corestore = require('corestore')

const TurboVecHybridAdapter =
  require('../src/adapters/database/TurboVecHybridAdapter')

// ----- args ---------------------------------------------------------------

function parseArgs (argv) {
  const out = { n: 3633, dim: 1024, queries: 100, k: 10, seed: 0xdeadbeef }
  for (const a of argv.slice(2)) {
    const m = /^--(\w+)=(.+)$/.exec(a)
    if (m) out[m[1]] = /^\d+$/.test(m[2]) ? Number(m[2]) : m[2]
  }
  return out
}

const args = parseArgs(Bare.argv)
const { n: N, dim: DIM, queries: NQ, k: K, seed: SEED } = args
const MODEL_ID = 'synthetic-1024'

// ----- helpers ------------------------------------------------------------

// xorshift32 for reproducible synthetic vectors. Not statistically
// rigorous — fine for "shape and timing" but don't infer recall from this.
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

function fmtMs (ms) { return ms.toFixed(2) + ' ms' }
function fmtMB (b)  { return (b / 1024 / 1024).toFixed(1) + ' MB' }

function rssMB () { return os.memoryUsage().rss / 1024 / 1024 }
function snap (label, t0, extra = '') {
  const dt = Date.now() - t0
  console.log(`  ${label.padEnd(28)} ${fmtMs(dt).padStart(12)}   rss=${rssMB().toFixed(1).padStart(6)} MB${extra ? '   ' + extra : ''}`)
  return dt
}

function percentile (sorted, p) {
  if (sorted.length === 0) return NaN
  const idx = Math.min(sorted.length - 1,
    Math.floor(sorted.length * (p / 100)))
  return sorted[idx]
}

// ----- main ---------------------------------------------------------------

async function main () {
  console.log('=== turbovec-perf (POC, naive scalar full-f32) ===')
  console.log(`N=${N} vectors, dim=${DIM}, queries=${NQ}, k=${K}, seed=0x${SEED.toString(16)}`)
  console.log(`baseline rss=${rssMB().toFixed(1)} MB`)
  console.log()

  const tmpRoot = path.join(os.tmpdir(),
    `turbovec-perf-${(os.pid && os.pid()) || 0}-${Date.now()}`)
  fs.mkdirSync(tmpRoot, { recursive: true })
  const indexPath = path.join(tmpRoot, 'perf.tvim')
  const storeDir = path.join(tmpRoot, 'corestore')

  // ----- generate vectors -----
  let t0 = Date.now()
  const rng = makePrng(SEED)
  const vectors = buildVecs(N, DIM, rng)
  snap('generate vectors', t0, `bytes=${fmtMB(vectors.byteLength)}`)
  const queryVecs = buildVecs(NQ, DIM, rng)

  // Build doc objects (adapter expects per-doc embedding arrays).
  t0 = Date.now()
  const docs = new Array(N)
  for (let i = 0; i < N; i++) {
    docs[i] = {
      id: `v${i}`,
      content: '',
      contentHash: '',
      embeddingModelId: MODEL_ID,
      embedding: Array.from(vectors.subarray(i * DIM, (i + 1) * DIM))
    }
  }
  snap('build doc objects', t0)

  // ----- adapter open -----
  const store = new Corestore(storeDir)
  const adapter = new TurboVecHybridAdapter({
    store,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  t0 = Date.now()
  await adapter.ready()
  snap('adapter.ready (empty)', t0)

  // ----- ingest -----
  t0 = Date.now()
  await adapter.saveEmbeddings(docs)
  const ingestMs = snap('saveEmbeddings (N total)', t0,
    `throughput=${(N / ((Date.now() - t0) / 1000) || N).toFixed(0)} vecs/s`)
  void ingestMs

  // ----- explicit re-persist (saveEmbeddings already writes the index;
  //       this is just to measure the standalone write cost) -----
  t0 = Date.now()
  await adapter.idx.write(indexPath)
  snap('idx.write (.tvim re-flush)', t0,
    `file=${fmtMB(fs.statSync(indexPath).size)}`)

  // ----- queries (top-k=K) -----
  // Two passes:
  //   (1) Per-query timing with Date.now (~4ms resolution on Bare) — gives
  //       distribution shape and tail behavior.
  //   (2) Bulk batch timing of NQ queries — gives a stable mean below the
  //       single-query resolution floor.
  const warmup = Math.min(5, NQ)
  for (let i = 0; i < warmup; i++) {
    const q = new Float32Array(queryVecs.subarray(i * DIM, (i + 1) * DIM))
    await adapter.search(null, q, { topK: K })
  }

  const latencies = []
  for (let i = 0; i < NQ; i++) {
    const q = new Float32Array(queryVecs.subarray((i % NQ) * DIM, (i % NQ) * DIM + DIM))
    const s = Date.now()
    await adapter.search(null, q, { topK: K })
    latencies.push(Date.now() - s)
  }
  latencies.sort((a, b) => a - b)
  const p50 = percentile(latencies, 50)
  const p95 = percentile(latencies, 95)
  const p99 = percentile(latencies, 99)

  // Bulk timing: same NQ queries, but measure the total wall-clock and
  // divide. Below the Date.now resolution this gives sub-ms accuracy.
  const bulkRepeat = Math.max(1, Math.ceil(200 / NQ))
  const bulkTotal = NQ * bulkRepeat
  const bulkStart = Date.now()
  for (let rep = 0; rep < bulkRepeat; rep++) {
    for (let i = 0; i < NQ; i++) {
      const q = new Float32Array(queryVecs.subarray(i * DIM, (i + 1) * DIM))
      await adapter.search(null, q, { topK: K })
    }
  }
  const bulkMs = Date.now() - bulkStart
  const bulkPer = bulkMs / bulkTotal

  console.log()
  console.log(`  query latency over ${NQ} samples (top-k=${K}):`)
  console.log(`    p50  = ${fmtMs(p50)}        (Date.now resolution ~4ms; use bulk-mean below for sub-ms)`)
  console.log(`    p95  = ${fmtMs(p95)}`)
  console.log(`    p99  = ${fmtMs(p99)}`)
  console.log(`    min  = ${fmtMs(latencies[0])}`)
  console.log(`    max  = ${fmtMs(latencies[latencies.length - 1])}`)
  console.log(`  bulk: ${bulkTotal} queries in ${bulkMs} ms => mean ${bulkPer.toFixed(3)} ms/query`)
  console.log()

  // ----- close + reopen -----
  await adapter.close()
  await store.close()

  t0 = Date.now()
  const store2 = new Corestore(storeDir)
  const adapter2 = new TurboVecHybridAdapter({
    store: store2,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  await adapter2.ready()
  snap('adapter.ready (reopen)', t0,
    `length=${adapter2.idx.length}`)

  // One post-reopen query to confirm the perf characteristic is steady.
  t0 = Date.now()
  await adapter2.search(null,
    new Float32Array(queryVecs.subarray(0, DIM)), { topK: K })
  snap('single query post-reopen', t0)

  await adapter2.close()
  await store2.close()
  fs.rmSync(tmpRoot, { recursive: true, force: true })

  console.log()
  console.log('--- done. POC scope: naive scalar full-f32; quantization + SIMD/GPU are the optimization phase ---')
}

main().catch((err) => {
  console.error(err)
  Bare.exit(1)
})

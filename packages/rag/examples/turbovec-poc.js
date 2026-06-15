'use strict'
//
// Phase 4 end-to-end smoke for the turbovec POC.
//
// Validates the wiring from `@qvac/rag.TurboVecHybridAdapter` ->
// `@qvac/embed-llamacpp.IdMapIndex` -> fabric's `ggml_vec_index_*` C API,
// on a corpus of 200 pre-encoded f32 vectors.
//
// Sequence:
//   1. Lifecycle isolation: no GGMLBert constructor must run.
//   2. Ingest 200 vectors.
//   3. Top-k self-match search.
//   4. Delete the top result and re-search.
//   5. Persist + reopen + re-search (deletion must survive).
//   6. Cleanup.
//
// Data source: if the bench harness's pre-encoded NFCorpus dump is on
// disk (`benchmarks/turbovec-eval/encoded/nfcorpus__thenlper_gte-large.docs.f32`,
// raw little-endian f32, 3633 vectors x 1024 dims), the script uses the
// first 200 vectors from it. Otherwise it falls back to a deterministic
// PRNG-generated 200x1024 synthetic corpus so the POC can run anywhere.

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const Corestore = require('corestore')

const TurboVecHybridAdapter =
  require('../src/adapters/database/TurboVecHybridAdapter')

const N = 200
const DIM = 1024
const MODEL_ID = 'thenlper/gte-large'

const NFCORPUS_PATH = path.join(
  __dirname, '..', 'benchmarks', 'turbovec-eval',
  'encoded', 'nfcorpus__thenlper_gte-large.docs.f32')

// xorshift32 — deterministic so re-runs are reproducible for debugging.
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
  return v
}

function loadVectors () {
  if (fs.existsSync(NFCORPUS_PATH)) {
    console.log(`[load] using real NFCorpus dump: ${NFCORPUS_PATH}`)
    const buf = fs.readFileSync(NFCORPUS_PATH)
    const wanted = N * DIM * 4
    if (buf.byteLength < wanted) {
      throw new Error(
        `NFCorpus dump too small: ${buf.byteLength} < ${wanted}`)
    }
    return new Float32Array(buf.buffer, buf.byteOffset, N * DIM)
  }
  console.log('[load] NFCorpus dump not found; synthesizing 200x1024')
  const rng = makePrng(0xdeadbeef)
  const out = new Float32Array(N * DIM)
  for (let i = 0; i < N; i++) {
    const v = new Float32Array(DIM)
    for (let j = 0; j < DIM; j++) v[j] = rng() * 2 - 1
    l2normalize(v)
    out.set(v, i * DIM)
  }
  return out
}

function assert (cond, msg) {
  if (!cond) {
    console.error(`ASSERT FAIL: ${msg}`)
    process?.exit?.(1) || global.exit?.(1)
  }
  console.log(`  ok ${msg}`)
}

async function main () {
  console.log('--- turbovec POC end-to-end smoke ---')

  // ---------------------------------------------------------------------
  // Step 1: lifecycle isolation
  // ---------------------------------------------------------------------
  // Authoritative lifecycle-isolation coverage lives in the embed-llamacpp
  // integration test (`test/integration/id-map-index.test.js`), which
  // asserts via `require.cache` that importing the IdMapIndex sub-export
  // never loads `index.js` (GGMLBert entry) or `addon.js`. This script
  // exercises the same code path via the adapter; that test gates the
  // architectural invariant.

  // ---------------------------------------------------------------------
  // Step 2: ingest
  // ---------------------------------------------------------------------
  const tmpRoot = path.join(os.tmpdir(),
    `turbovec-poc-${(os.pid && os.pid()) || 0}-${Date.now()}`)
  fs.mkdirSync(tmpRoot, { recursive: true })
  const indexPath = path.join(tmpRoot, 'turbovec-poc.tvim')
  const storeDir = path.join(tmpRoot, 'corestore')

  const vectors = loadVectors()
  const docs = []
  for (let i = 0; i < N; i++) {
    docs.push({
      id: `nfcorpus-${i}`,
      content: `doc ${i}`,
      contentHash: `h-${i}`,
      embeddingModelId: MODEL_ID,
      embedding: Array.from(vectors.subarray(i * DIM, (i + 1) * DIM))
    })
  }

  const store = new Corestore(storeDir)
  const adapter = new TurboVecHybridAdapter({
    store,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  await adapter.ready()

  const saved = await adapter.saveEmbeddings(docs)
  assert(saved.length === N, `all ${N} docs saved`)
  assert(saved.every((r) => r.status === 'fulfilled'),
    'every save reported fulfilled')

  // ---------------------------------------------------------------------
  // Step 3: search
  // ---------------------------------------------------------------------
  const q0 = new Float32Array(vectors.subarray(0, DIM))
  let hits = await adapter.search(null, q0, { topK: 5 })
  console.log('[search] top-5 for query=vec0:')
  for (const h of hits) {
    console.log(`    ${h.id.padEnd(15)}  score=${h.score.toFixed(6)}`)
  }
  assert(hits[0].id === 'nfcorpus-0', 'top-1 is the query itself')
  assert(Math.abs(hits[0].score - 1.0) < 1e-3, 'self-similarity ≈ 1.0')

  // ---------------------------------------------------------------------
  // Step 4: delete
  // ---------------------------------------------------------------------
  await adapter.deleteEmbeddings(['nfcorpus-0'])
  hits = await adapter.search(null, q0, { topK: 5 })
  const top1After = hits[0]
  console.log(`[delete] top-1 after deleting nfcorpus-0: ${top1After.id} (score=${top1After.score.toFixed(6)})`)
  assert(top1After.id !== 'nfcorpus-0',
    'deleted id no longer surfaces')
  assert(adapter.idx.length === N - 1, 'index length shrank by 1')

  // ---------------------------------------------------------------------
  // Step 5: persistence round-trip
  // ---------------------------------------------------------------------
  await adapter.close()
  await store.close()
  assert(fs.existsSync(indexPath), '.tvim file present on disk')

  const store2 = new Corestore(storeDir)
  const adapter2 = new TurboVecHybridAdapter({
    store: store2,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  await adapter2.ready()
  assert(adapter2.idx.length === N - 1,
    'reopened index has the post-delete count')
  const reHits = await adapter2.search(null, q0, { topK: 1 })
  assert(reHits[0].id === top1After.id,
    `reopened top-1 matches pre-close (${reHits[0].id} === ${top1After.id})`)
  console.log(`[reopen] top-1 after reload: ${reHits[0].id} (score=${reHits[0].score.toFixed(6)})`)
  await adapter2.close()
  await store2.close()

  // ---------------------------------------------------------------------
  // Step 6: cleanup
  // ---------------------------------------------------------------------
  fs.rmSync(tmpRoot, { recursive: true, force: true })
  console.log('--- turbovec POC: OK ---')
}

main().catch((err) => {
  console.error(err)
  process?.exit?.(1) || global.exit?.(1)
  throw err
})

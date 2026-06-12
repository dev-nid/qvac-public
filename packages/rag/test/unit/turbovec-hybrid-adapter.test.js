'use strict'
//
// Phase 3 acceptance test (turbovec POC).
// Exercises TurboVecHybridAdapter against a temporary Corestore:
//   construct -> save 5 -> search -> delete one -> search -> close -> reopen -> search.

const test = require('brittle')
const tmp = require('test-tmp')
const path = require('bare-path')
const fs = require('bare-fs')
const Corestore = require('corestore')

const TurboVecHybridAdapter =
  require('../../src/adapters/database/TurboVecHybridAdapter')
const stringIdToU64 =
  require('../../src/utils/stringIdToU64')

const DIM = 8
const MODEL_ID = 'gte-small-test'

// Build N seed unit vectors in dim DIM (one per axis up to DIM).
function seeds (n) {
  const out = []
  for (let i = 0; i < n; i++) {
    const v = new Float32Array(DIM)
    v[i % DIM] = 1
    out.push(Array.from(v))
  }
  return out
}

function makeDocs (n) {
  const v = seeds(n)
  return Array.from({ length: n }, (_, i) => ({
    id: `doc-${i}`,
    content: `document number ${i}`,
    contentHash: `hash-${i}`,
    embeddingModelId: MODEL_ID,
    embedding: v[i],
    metadata: { axis: i }
  }))
}

test('stringIdToU64: deterministic and well-distributed', (t) => {
  const a1 = stringIdToU64('doc-1')
  const a2 = stringIdToU64('doc-1')
  t.is(a1, a2, 'deterministic')
  t.unlike(a1, stringIdToU64('doc-2'), 'different strings differ')

  let threw = false
  try { stringIdToU64('') } catch (_) { threw = true }
  t.ok(threw, 'rejects empty')

  threw = false
  try { stringIdToU64(42) } catch (_) { threw = true }
  t.ok(threw, 'rejects non-string')
})

test('TurboVecHybridAdapter: save -> search -> delete -> reopen -> search', async (t) => {
  const tmpDir = await tmp()
  const indexPath = path.join(tmpDir, 'test.tvim')

  const store1 = new Corestore(tmpDir)
  const adapter1 = new TurboVecHybridAdapter({
    store: store1,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  await adapter1.ready()
  t.is(adapter1.idx.length, 0, 'starts empty')

  const docs = makeDocs(5)
  const results = await adapter1.saveEmbeddings(docs)
  t.is(results.length, 5, 'all 5 docs reported as fulfilled')
  for (const r of results) {
    t.is(r.status, 'fulfilled')
  }
  t.is(adapter1.idx.length, 5, 'index has all 5 vectors')

  // Search: query with vector axis-0 must retrieve doc-0 as top result.
  {
    const hits = await adapter1.search(null, new Float32Array(seeds(1)[0]),
      { topK: 3 })
    t.is(hits.length, 3, 'returned 3 hits')
    t.is(hits[0].id, 'doc-0', 'top-1 is doc-0 (self)')
    t.ok(Math.abs(hits[0].score - 1.0) < 1e-5, 'score≈1.0')
    t.is(hits[0].content, 'document number 0', 'content surfaced')
    t.is(hits[0].metadata.axis, 0, 'metadata round-trips')
  }

  // Delete: removing doc-0 means a subsequent search for axis-0 should
  // surface another doc (not doc-0).
  await adapter1.deleteEmbeddings(['doc-0'])
  t.is(adapter1.idx.length, 4, 'index size shrank by 1')
  {
    const hits = await adapter1.search(null, new Float32Array(seeds(1)[0]),
      { topK: 3 })
    t.absent(hits.some((h) => h.id === 'doc-0'),
      'doc-0 absent from results after delete')
  }

  await adapter1.close()
  // The HyperDB.bee() inside the adapter shares the hypercore with the
  // store; close the corestore too so the rocksdb FD lock can be reacquired
  // by the reopen-step Corestore below.
  await store1.close()
  t.ok(fs.existsSync(indexPath), '.tvim file written on close')

  // Reopen: persistence survived close+reopen.
  const store2 = new Corestore(tmpDir)
  const adapter2 = new TurboVecHybridAdapter({
    store: store2,
    dim: DIM,
    bitWidth: 4,
    indexPath,
    embeddingModelId: MODEL_ID
  })
  await adapter2.ready()
  t.is(adapter2.idx.length, 4, 'index size restored from .tvim')
  // Self-match for doc-1 (axis-1) must still be doc-1.
  {
    const hits = await adapter2.search(null, new Float32Array(seeds(2)[1]),
      { topK: 1 })
    t.is(hits.length, 1)
    t.is(hits[0].id, 'doc-1', 'self-match survives reopen')
  }
  // Reverse map was rebuilt: deleted doc-0 stays absent.
  t.absent(adapter2._u64ToStringId.has(stringIdToU64('doc-0')),
    'reverse-map omits deleted doc')

  // reindex() is a no-op.
  const reindexResult = await adapter2.reindex()
  t.is(reindexResult.reindexed, false)

  await adapter2.close()
  await store2.close()
})

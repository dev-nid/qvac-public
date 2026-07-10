'use strict'
//
// Phase 2 acceptance: exercises every JS method on `IdMapIndex` end-to-end
// without loading any BERT model. Pair-bound with the C++ test under
// fabric's `tests/test-vector-index.cpp` (which exercises the same C API
// directly): if both pass, the JS↔C++ binding is correct in both
// directions.

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')

// Pull IdMapIndex via the package's sub-export path. The package exports
// `./idMapIndex` so consumers can opt into just the ANN-index class
// without dragging the GGMLBert require chain. Loading this module must
// NOT boot any BERT runtime. Resolved via a relative path here because
// the integration tests run inside the package itself (no self-link).
//
// To catch a future regression that wires GGMLBert init back into the
// IdMapIndex path, the first test below inspects `require.cache` and
// asserts the GGMLBert module file was never loaded. Bare keys
// `require.cache` by `file://` URL, so we convert the resolved paths.
const cacheKey = (p) => 'file://' + require.resolve(p)
const KEY_IDMAP   = cacheKey('../../idMapIndex')
const KEY_BINDING = cacheKey('../../binding')
const KEY_INDEX   = cacheKey('../../index.js')
const KEY_ADDON   = cacheKey('../../addon.js')

const IdMapIndex = require('../../idMapIndex')

// DIM is chosen large enough that the N seed vectors below can each be a
// distinct unit basis vector — keeps the "self-match top-1" assertion
// well-defined.
const DIM = 16
const N = 10

function unitVec (i) {
  const v = new Float32Array(DIM)
  v[i] = 1
  return v
}

let tmpCounter = 0
function tmpPath (name) {
  const pid = os.pid ? os.pid() : 0
  tmpCounter += 1
  return path.join(os.tmpdir(),
    `${name}-${pid}-${Date.now()}-${tmpCounter}.tvim`)
}

test('IdMapIndex sub-export does not boot the BERT runtime', (t) => {
  // 1. Construct path: zero-arg + tiny-arg ctors must succeed without
  //    loading any model or running addon-side BERT init.
  const idx = new IdMapIndex({ dim: DIM, bitWidth: 4 })
  t.is(idx.dim, DIM, 'dim getter')
  t.is(idx.bitWidth, 4, 'bitWidth getter')
  t.is(idx.length, 0, 'starts empty')
  idx.dispose()

  // 2. Module-cache invariant: requiring `./idMapIndex` must NOT have
  //    transitively loaded the GGMLBert entry (`./index.js`) or the
  //    BertInterface plumbing (`./addon.js`). If a future refactor wires
  //    BERT init into the IdMapIndex path, one of these files will be
  //    in the cache after the require above.
  t.ok(require.cache[KEY_IDMAP],
    'sub-export module loaded (sanity)')
  t.ok(require.cache[KEY_BINDING],
    'native binding loaded (sanity)')
  t.absent(require.cache[KEY_INDEX],
    'GGMLBert entry (index.js) NOT loaded by ./idMapIndex')
  t.absent(require.cache[KEY_ADDON],
    'BertInterface plumbing (addon.js) NOT loaded by ./idMapIndex')
})

test('IdMapIndex: add + search + remove + persistence round-trip', async (t) => {
  const idx = new IdMapIndex({ dim: DIM, bitWidth: 4 })

  const vectors = new Float32Array(N * DIM)
  const ids = new BigUint64Array(N)
  for (let i = 0; i < N; i++) {
    vectors.set(unitVec(i), i * DIM)
    // Mix non-trivial ids to flush out BigInt/uint64 round-trip bugs.
    ids[i] = BigInt(i) + (1n << 40n) + (1n << 62n)
  }

  idx.addWithIds(vectors, ids)
  t.is(idx.length, N, 'all 10 vectors inserted')
  t.ok(idx.contains(ids[3]), 'contains a known id')
  t.absent(idx.contains(999n), 'absent id missing')

  // Top-1 of querying with the i-th unit vector must return that vector's id
  // with a score very close to 1.0 (full f32, no quantization noise).
  for (let i = 0; i < N; i++) {
    const q = unitVec(i)
    const out = idx.search(q, 1)
    t.is(out.m, 1)
    t.is(out.k, 1)
    t.is(out.ids[0], ids[i], `query=${i} retrieves itself`)
    t.ok(Math.abs(out.scores[0] - 1.0) < 1e-5, `score≈1.0 for self-match`)
  }

  // Top-k > length pads sentinels at the tail.
  {
    const out = idx.search(unitVec(0), N + 4)
    t.is(out.ids.length, N + 4)
    for (let i = N; i < N + 4; i++) {
      t.is(out.ids[i], (1n << 64n) - 1n, `sentinel id at tail slot ${i}`)
    }
  }

  // Duplicate add must throw atomically (length unchanged).
  {
    const dupIds = new BigUint64Array([ids[0]])
    const dupVec = unitVec(0)
    try {
      idx.addWithIds(dupVec, dupIds)
      t.fail('duplicate add should have thrown')
    } catch (e) {
      t.pass('duplicate add threw: ' + (e.message || e.code))
    }
    t.is(idx.length, N, 'length unchanged after rejected dup add')
  }

  // Remove + search: the removed id should not surface.
  {
    const removed = idx.remove(ids[2])
    t.ok(removed === true, 'remove returns true on first call')
    t.absent(idx.remove(ids[2]), 'remove returns false on second call')
    t.absent(idx.contains(ids[2]), 'no longer contained')
    t.is(idx.length, N - 1)
    const out = idx.search(unitVec(2), 3)
    for (let i = 0; i < 3; i++) {
      t.absent(out.ids[i] === ids[2], `removed id absent from search result ${i}`)
    }
  }

  // prepare() is a no-op but must be callable.
  idx.prepare()

  // Persist, dispose, reload, re-search.
  const file = tmpPath('id-map-index-roundtrip')
  try {
    idx.write(file)
    await idx.dispose()

    const loaded = await IdMapIndex.load(file)
    t.is(loaded.dim, DIM, 'dim restored')
    t.is(loaded.bitWidth, 4, 'bitWidth restored')
    t.is(loaded.length, N - 1, 'length restored (deletion persisted)')
    t.ok(loaded.contains(ids[0]), 'kept id still there')
    t.absent(loaded.contains(ids[2]), 'deleted id stayed deleted')

    const out = loaded.search(unitVec(0), 1)
    t.is(out.ids[0], ids[0], 'self-match after reload')
    t.ok(Math.abs(out.scores[0] - 1.0) < 1e-5)

    await loaded.dispose()
  } finally {
    if (fs.existsSync(file)) fs.unlinkSync(file)
  }
})

test('IdMapIndex: BigInt id range edge cases', (t) => {
  // 2^63 + 7 catches sign-extension bugs at the typed-array boundary.
  const idx = new IdMapIndex({ dim: 2 })
  const edge = (1n << 63n) + 7n
  idx.addWithIds(new Float32Array([0.5, 0.5]), new BigUint64Array([edge]))
  t.ok(idx.contains(edge), 'high-bit id round-trips')
  const out = idx.search(new Float32Array([0.5, 0.5]), 1)
  t.is(out.ids[0], edge, 'high-bit id surfaces from search')
  idx.dispose()
})

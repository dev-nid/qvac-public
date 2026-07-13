'use strict'
//
// Exercises the production IdMapIndex JS surface end-to-end without loading
// any BERT model. Fabric's own tests cover the C API directly; these tests
// focus on the JS/native binding contract and packaging path.

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
const KEY_IDMAP = cacheKey('../../idMapIndex')
const KEY_BINDING = cacheKey('../../binding')
const KEY_INDEX = cacheKey('../../index.js')
const KEY_ADDON = cacheKey('../../addon.js')

const IdMapIndex = require('../../idMapIndex')

// DIM is chosen large enough that the N seed vectors below can each be a
// distinct unit basis vector — keeps the "self-match top-1" assertion
// well-defined.
const DIM = 16
const N = 10
const UINT64_MAX = (1n << 64n) - 1n

function unitVec(i) {
  const v = new Float32Array(DIM)
  v[i] = 1
  return v
}

let tmpCounter = 0
function tmpPath(name) {
  const pid = os.pid ? os.pid() : 0
  tmpCounter += 1
  return path.join(os.tmpdir(), `${name}-${pid}-${Date.now()}-${tmpCounter}.tvim`)
}

function expectThrows(t, fn, message) {
  try {
    fn()
    t.fail(message)
  } catch (e) {
    t.pass(`${message}: ${e.message || e.code}`)
  }
}

async function expectRejects(t, fn, message) {
  try {
    await fn()
    t.fail(message)
  } catch (e) {
    t.pass(`${message}: ${e.message || e.code}`)
  }
}

function assertTvimV2Header(t, file, bitWidth) {
  const bytes = fs.readFileSync(file)
  t.is(bytes[0], 0x54, 'tvim magic T')
  t.is(bytes[1], 0x56, 'tvim magic V')
  t.is(bytes[2], 0x50, 'tvim magic P')
  t.is(bytes[3], 0x49, 'tvim magic I')
  t.is(bytes[4], 2, 'tvim version 2')
  t.is(bytes[5], bitWidth, 'tvim bit width')
}

async function runRoundTrip(t, bitWidth) {
  const idx = new IdMapIndex({ dim: DIM, bitWidth })

  const vectors = new Float32Array(N * DIM)
  const ids = new BigUint64Array(N)
  for (let i = 0; i < N; i++) {
    vectors.set(unitVec(i), i * DIM)
    // Mix non-trivial ids to flush out BigInt/uint64 round-trip bugs.
    ids[i] = BigInt(i) + (1n << 40n) + (1n << 62n)
  }

  idx.addWithIds(vectors, ids)
  t.is(idx.length, N, `all vectors inserted (${bitWidth})`)
  t.ok(idx.contains(ids[3]), `contains a known id (${bitWidth})`)
  t.absent(idx.contains(999n), `absent id missing (${bitWidth})`)

  // Unit vectors remain exact for both f32 and q8 storage.
  for (let i = 0; i < N; i++) {
    const out = idx.search(unitVec(i), 1)
    t.is(out.m, 1)
    t.is(out.k, 1)
    t.is(out.ids[0], ids[i], `query=${i} retrieves itself (${bitWidth})`)
    t.ok(Math.abs(out.scores[0] - 1.0) < 1e-5, `score≈1.0 for self-match (${bitWidth})`)
  }

  {
    const out = idx.search(unitVec(0), N + 4)
    t.is(out.ids.length, N + 4)
    t.is(out.scores.length, N + 4)
    for (let i = N; i < N + 4; i++) {
      t.is(out.ids[i], UINT64_MAX, `sentinel id at tail slot ${i} (${bitWidth})`)
      t.ok(out.scores[i] < -3e38, `sentinel score at tail slot ${i} (${bitWidth})`)
    }
  }

  {
    const dupIds = new BigUint64Array([ids[0]])
    expectThrows(
      t,
      () => idx.addWithIds(unitVec(0), dupIds),
      `duplicate add should throw (${bitWidth})`
    )
    t.is(idx.length, N, `length unchanged after rejected dup add (${bitWidth})`)
  }

  {
    const removed = idx.remove(ids[2])
    t.ok(removed === true, `remove returns true on first call (${bitWidth})`)
    t.absent(idx.remove(ids[2]), `remove returns false on second call (${bitWidth})`)
    t.absent(idx.contains(ids[2]), `no longer contained (${bitWidth})`)
    t.is(idx.length, N - 1)
    const out = idx.search(unitVec(2), 3)
    for (let i = 0; i < 3; i++) {
      t.absent(out.ids[i] === ids[2], `removed id absent from search result ${i} (${bitWidth})`)
    }
  }

  idx.prepare()

  const file = tmpPath(`id-map-index-roundtrip-${bitWidth}`)
  try {
    idx.write(file)
    assertTvimV2Header(t, file, bitWidth)
    await idx.dispose()

    const loaded = await IdMapIndex.load(file)
    t.is(loaded.dim, DIM, `dim restored (${bitWidth})`)
    t.is(loaded.bitWidth, bitWidth, `bitWidth restored (${bitWidth})`)
    t.is(loaded.length, N - 1, `length restored (${bitWidth})`)
    t.ok(loaded.contains(ids[0]), `kept id still there (${bitWidth})`)
    t.absent(loaded.contains(ids[2]), `deleted id stayed deleted (${bitWidth})`)

    const out = loaded.search(unitVec(0), 1)
    t.is(out.ids[0], ids[0], `self-match after reload (${bitWidth})`)
    t.ok(Math.abs(out.scores[0] - 1.0) < 1e-5, `score≈1.0 after reload (${bitWidth})`)

    await loaded.dispose()
  } finally {
    if (fs.existsSync(file)) fs.unlinkSync(file)
  }
}

test('IdMapIndex sub-export does not boot the BERT runtime', (t) => {
  // 1. Construct path: zero-arg + tiny-arg ctors must succeed without
  //    loading any model or running addon-side BERT init.
  const idx = new IdMapIndex({ dim: DIM })
  t.is(idx.dim, DIM, 'dim getter')
  t.is(idx.bitWidth, 8, 'default bitWidth getter')
  t.is(idx.length, 0, 'starts empty')
  idx.dispose()

  // 2. Module-cache invariant: requiring `./idMapIndex` must NOT have
  //    transitively loaded the GGMLBert entry (`./index.js`) or the
  //    BertInterface plumbing (`./addon.js`). If a future refactor wires
  //    BERT init into the IdMapIndex path, one of these files will be
  //    in the cache after the require above.
  t.ok(require.cache[KEY_IDMAP], 'sub-export module loaded (sanity)')
  t.ok(require.cache[KEY_BINDING], 'native binding loaded (sanity)')
  t.absent(require.cache[KEY_INDEX], 'GGMLBert entry (index.js) NOT loaded by ./idMapIndex')
  t.absent(require.cache[KEY_ADDON], 'BertInterface plumbing (addon.js) NOT loaded by ./idMapIndex')
})

test('IdMapIndex: q8 add + search + remove + persistence round-trip', async (t) => {
  await runRoundTrip(t, 8)
})

test('IdMapIndex: f32 add + search + remove + persistence round-trip', async (t) => {
  await runRoundTrip(t, 32)
})

test('IdMapIndex: validates production bit widths', (t) => {
  expectThrows(t, () => new IdMapIndex({ dim: DIM, bitWidth: 4 }), 'bitWidth 4 should be rejected')
  expectThrows(
    t,
    () => new IdMapIndex({ dim: DIM, bitWidth: 16 }),
    'bitWidth 16 should be rejected'
  )
})

test('IdMapIndex: rejects mismatched empty-id add', (t) => {
  const idx = new IdMapIndex({ dim: 2 })
  expectThrows(
    t,
    () => {
      idx.addWithIds(new Float32Array([1, 2]), new BigUint64Array())
    },
    'vectors with empty ids should be rejected'
  )
  t.is(idx.length, 0, 'failed add does not mutate')
  idx.addWithIds(new Float32Array(), new BigUint64Array())
  t.is(idx.length, 0, 'empty add remains a no-op')
  idx.dispose()
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

test('IdMapIndex: rejects non-finite vectors and queries', (t) => {
  const idx = new IdMapIndex({ dim: 2, bitWidth: 8 })
  expectThrows(
    t,
    () => {
      idx.addWithIds(new Float32Array([1, Number.POSITIVE_INFINITY]), new BigUint64Array([1n]))
    },
    'non-finite vector should be rejected'
  )
  t.is(idx.length, 0, 'failed add does not mutate')

  idx.addWithIds(new Float32Array([1, 0]), new BigUint64Array([1n]))
  expectThrows(
    t,
    () => idx.search(new Float32Array([Number.NaN, 0]), 1),
    'non-finite query should be rejected'
  )
  idx.dispose()
})

test('IdMapIndex: corrupt persistence file load fails', async (t) => {
  const file = tmpPath('id-map-index-corrupt')
  try {
    fs.writeFileSync(file, new Uint8Array([0, 1, 2, 3, 4, 5]))
    await expectRejects(t, () => IdMapIndex.load(file), 'corrupt tvim file should be rejected')
  } finally {
    if (fs.existsSync(file)) fs.unlinkSync(file)
  }
})

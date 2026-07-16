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

function tmpDeltaPath(name) {
  return tmpPath(name).replace(/\.tvim$/, '.tvid')
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

function assertTvidHeader(t, file) {
  const bytes = fs.readFileSync(file)
  t.is(bytes[0], 0x54, 'tvid magic T')
  t.is(bytes[1], 0x56, 'tvid magic V')
  t.is(bytes[2], 0x44, 'tvid magic D')
  t.is(bytes[3], 0x4c, 'tvid magic L')
  t.is(bytes[4], 1, 'tvid version 1')
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

  // Unit vectors remain exact for f32, q8, and q4 storage.
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

  idx.compact()
  t.is(idx.length, N - 1, `length preserved after compact (${bitWidth})`)
  t.absent(idx.contains(ids[2]), `removed id absent after compact (${bitWidth})`)
  t.is(idx.search(unitVec(0), 1).ids[0], ids[0], `search works after compact (${bitWidth})`)

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

test('IdMapIndex: q4 add + search + remove + persistence round-trip', async (t) => {
  await runRoundTrip(t, 4)
})

test('IdMapIndex: q8 add + search + remove + persistence round-trip', async (t) => {
  await runRoundTrip(t, 8)
})

test('IdMapIndex: f32 add + search + remove + persistence round-trip', async (t) => {
  await runRoundTrip(t, 32)
})

test('IdMapIndex: validates production bit widths', (t) => {
  expectThrows(
    t,
    () => new IdMapIndex({ dim: DIM, bitWidth: 16 }),
    'bitWidth 16 should be rejected'
  )
})

test('IdMapIndex: promise APIs reject instead of throwing synchronously', async (t) => {
  const load = IdMapIndex.load('')
  t.ok(load && typeof load.catch === 'function', 'load returns a promise for invalid input')
  await expectRejects(t, () => load, 'load invalid path rejects')

  const loadMmap = IdMapIndex.loadMmap('')
  t.ok(
    loadMmap && typeof loadMmap.catch === 'function',
    'loadMmap returns a promise for invalid input'
  )
  await expectRejects(t, () => loadMmap, 'loadMmap invalid path rejects')

  const loadWithDelta = IdMapIndex.loadWithDelta('', '')
  t.ok(
    loadWithDelta && typeof loadWithDelta.catch === 'function',
    'loadWithDelta returns a promise for invalid input'
  )
  await expectRejects(t, () => loadWithDelta, 'loadWithDelta invalid paths reject')
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

test('IdMapIndex: filtered search restricts allowed ids', (t) => {
  const idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
  idx.addWithIds(new Float32Array([1, 0, 0, 1, 0.5, 0.5]), new BigUint64Array([11n, 22n, 33n]))

  {
    const out = idx.searchFiltered(new Float32Array([1, 0]), 2, new BigUint64Array([22n, 33n, 44n]))
    t.is(out.m, 1, 'filtered m')
    t.is(out.k, 2, 'filtered k')
    t.is(out.ids[0], 33n, 'best allowed id wins')
    t.is(out.ids[1], 22n, 'lower-scoring allowed id follows')
  }

  {
    const out = idx.searchFiltered(new Float32Array([1, 0]), 2, new BigUint64Array())
    t.is(out.ids[0], UINT64_MAX, 'empty filter returns sentinel id 0')
    t.ok(out.scores[0] < -3e38, 'empty filter returns sentinel score 0')
    t.is(out.ids[1], UINT64_MAX, 'empty filter returns sentinel id 1')
    t.ok(out.scores[1] < -3e38, 'empty filter returns sentinel score 1')
  }

  idx.dispose()
})

test('IdMapIndex: prepared filters are reusable and invalidated by mutation', async (t) => {
  const idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
  let filter = null
  try {
    idx.addWithIds(new Float32Array([1, 0, 0, 1, 0.5, 0.5]), new BigUint64Array([11n, 22n, 33n]))
    filter = idx.prepareFilter(new BigUint64Array([22n, 33n]))

    const first = filter.search(new Float32Array([1, 0]), 2)
    t.is(first.ids[0], 33n, 'prepared filter best allowed id wins')
    t.is(first.ids[1], 22n, 'prepared filter lower allowed id follows')

    const second = filter.search(new Float32Array([0, 1]), 1)
    t.is(second.ids[0], 22n, 'prepared filter can be reused')

    idx.addWithIds(new Float32Array([0.25, 0.75]), new BigUint64Array([44n]))
    expectThrows(
      t,
      () => filter.search(new Float32Array([1, 0]), 1),
      'stale prepared filter should throw'
    )

    await filter.dispose()
    await filter.dispose()
    expectThrows(
      t,
      () => filter.search(new Float32Array([1, 0]), 1),
      'disposed prepared filter should throw'
    )
    filter = null
  } finally {
    if (filter !== null) await filter.dispose()
    await idx.dispose()
  }
})

test('IdMapIndex: IVF build and search lifecycle', (t) => {
  const idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
  idx.addWithIds(
    new Float32Array([1, 0, 0, 1, 0.5, 0.5, -1, 0]),
    new BigUint64Array([11n, 22n, 33n, 44n])
  )

  expectThrows(
    t,
    () => idx.searchIvf(new Float32Array([1, 0]), 1, 1),
    'IVF search before build should throw'
  )

  idx.buildIvf(4, 0)
  const out = idx.searchIvf(new Float32Array([1, 0]), 2, 4)
  t.is(out.m, 1, 'IVF m')
  t.is(out.k, 2, 'IVF k')
  t.is(out.ids[0], 11n, 'IVF search returns nearest id')

  idx.addWithIds(new Float32Array([0.25, 0.75]), new BigUint64Array([55n]))
  expectThrows(
    t,
    () => idx.searchIvf(new Float32Array([1, 0]), 1, 4),
    'IVF search after mutation should throw'
  )

  idx.buildIvf(4, 1)
  t.is(idx.searchIvf(new Float32Array([0, 1]), 1, 4).ids[0], 22n, 'IVF rebuild restores search')
  idx.dispose()
})

test('IdMapIndex: delta log replay and compaction', async (t) => {
  const snapshot = tmpPath('id-map-index-delta-snapshot')
  const delta = tmpDeltaPath('id-map-index-delta-log')
  let idx = null
  let replayed = null
  let compacted = null
  try {
    idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
    idx.addWithIds(new Float32Array([1, 0, 0, 1]), new BigUint64Array([11n, 22n]))
    idx.write(snapshot)

    idx.addLogged(new Float32Array([0.5, 0.5]), new BigUint64Array([33n]), delta)
    t.ok(idx.removeLogged(11n, delta), 'logged remove returns true for existing id')
    t.absent(idx.removeLogged(44n, delta), 'logged remove returns false for absent id')
    t.is(idx.length, 2, 'logged mutations update live index')

    replayed = await IdMapIndex.loadWithDelta(snapshot, delta)
    t.is(replayed.dim, 2, 'delta replay dim restored')
    t.is(replayed.bitWidth, 4, 'delta replay bitWidth restored')
    t.absent(replayed.contains(11n), 'delta replay applies remove')
    t.ok(replayed.contains(22n), 'delta replay keeps snapshot id')
    t.ok(replayed.contains(33n), 'delta replay applies add')
    t.is(
      replayed.searchFiltered(new Float32Array([0.5, 0.5]), 1, new BigUint64Array([33n])).ids[0],
      33n,
      'delta replay search sees added id'
    )
    await replayed.dispose()
    replayed = null

    idx.compactDelta(snapshot, delta)
    assertTvimV2Header(t, snapshot, 4)
    assertTvidHeader(t, delta)
    await idx.dispose()
    idx = null

    compacted = await IdMapIndex.loadWithDelta(snapshot, delta)
    t.absent(compacted.contains(11n), 'compacted snapshot excludes removed id')
    t.ok(compacted.contains(22n), 'compacted snapshot keeps existing id')
    t.ok(compacted.contains(33n), 'compacted snapshot includes logged add')
  } finally {
    if (idx !== null) await idx.dispose()
    if (replayed !== null) await replayed.dispose()
    if (compacted !== null) await compacted.dispose()
    if (fs.existsSync(snapshot)) fs.unlinkSync(snapshot)
    if (fs.existsSync(delta)) fs.unlinkSync(delta)
  }
})

test('IdMapIndex: missing delta log replays as empty', async (t) => {
  const snapshot = tmpPath('id-map-index-missing-delta-snapshot')
  const delta = tmpDeltaPath('id-map-index-missing-delta-log')
  let idx = null
  let loaded = null
  try {
    idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
    idx.addWithIds(new Float32Array([1, 0, 0, 1]), new BigUint64Array([11n, 22n]))
    idx.write(snapshot)
    await idx.dispose()
    idx = null

    t.absent(fs.existsSync(delta), 'delta log does not exist before load')
    loaded = await IdMapIndex.loadWithDelta(snapshot, delta)
    t.is(loaded.dim, 2, 'snapshot dim restored without delta')
    t.is(loaded.bitWidth, 4, 'snapshot bitWidth restored without delta')
    t.is(loaded.length, 2, 'snapshot length restored without delta')
    t.ok(loaded.contains(11n), 'snapshot id 11 restored without delta')
    t.ok(loaded.contains(22n), 'snapshot id 22 restored without delta')
  } finally {
    if (idx !== null) await idx.dispose()
    if (loaded !== null) await loaded.dispose()
    if (fs.existsSync(snapshot)) fs.unlinkSync(snapshot)
    if (fs.existsSync(delta)) fs.unlinkSync(delta)
  }
})

test('IdMapIndex: corrupt delta log replay fails', async (t) => {
  const snapshot = tmpPath('id-map-index-corrupt-delta-snapshot')
  const delta = tmpDeltaPath('id-map-index-corrupt-delta-log')
  let idx = null
  try {
    idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
    idx.addWithIds(new Float32Array([1, 0]), new BigUint64Array([11n]))
    idx.write(snapshot)
    await idx.dispose()
    idx = null

    fs.writeFileSync(delta, new Uint8Array([0, 1, 2, 3, 4, 5]))
    await expectRejects(
      t,
      () => IdMapIndex.loadWithDelta(snapshot, delta),
      'corrupt delta log should be rejected'
    )
  } finally {
    if (idx !== null) await idx.dispose()
    if (fs.existsSync(snapshot)) fs.unlinkSync(snapshot)
    if (fs.existsSync(delta)) fs.unlinkSync(delta)
  }
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

test('IdMapIndex: mmap load is searchable and read-only', async (t) => {
  const file = tmpPath('id-map-index-mmap')
  let mmap = null
  let filter = null
  try {
    const idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
    idx.addWithIds(new Float32Array([1, 0, 0, 1]), new BigUint64Array([11n, 22n]))
    idx.write(file)
    await idx.dispose()

    mmap = await IdMapIndex.loadMmap(file)
    t.is(mmap.dim, 2, 'mmap dim restored')
    t.is(mmap.bitWidth, 4, 'mmap bitWidth restored')
    t.is(mmap.length, 2, 'mmap length restored')
    t.ok(mmap.contains(22n), 'mmap contains existing id')
    t.is(mmap.search(new Float32Array([0, 1]), 1).ids[0], 22n, 'mmap search works')
    t.is(
      mmap.searchFiltered(new Float32Array([1, 0]), 1, new BigUint64Array([11n])).ids[0],
      11n,
      'mmap filtered search works'
    )
    filter = mmap.prepareFilter(new BigUint64Array([11n]))
    t.is(
      filter.search(new Float32Array([1, 0]), 1).ids[0],
      11n,
      'mmap prepared filter search works'
    )

    expectThrows(
      t,
      () => mmap.addWithIds(new Float32Array([0.5, 0.5]), new BigUint64Array([33n])),
      'mmap add should be rejected'
    )
    expectThrows(t, () => mmap.remove(11n), 'mmap remove should be rejected')
    expectThrows(t, () => mmap.compact(), 'mmap compact should be rejected')
  } finally {
    if (filter !== null) await filter.dispose()
    if (mmap !== null) await mmap.dispose()
    if (fs.existsSync(file)) fs.unlinkSync(file)
  }
})

test('IdMapIndex: dispose is deterministic and idempotent', async (t) => {
  const idx = new IdMapIndex({ dim: 2, bitWidth: 4 })
  idx.addWithIds(new Float32Array([1, 0]), new BigUint64Array([1n]))
  await idx.dispose()
  await idx.dispose()

  expectThrows(t, () => idx.contains(1n), 'disposed contains should throw')
  expectThrows(t, () => idx.search(new Float32Array([1, 0]), 1), 'disposed search should throw')
  expectThrows(t, () => idx.compact(), 'disposed compact should throw')
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

'use strict'
//
// TurboVecHybridAdapter
// =====================
//
// Hybrid database adapter for the turbovec POC. Uses HyperDB for document
// rows + replication (same `@rag/documents` and `@rag/config` schema as
// HyperDBAdapter so corestores are interchangeable for read paths) and
// the fabric-native `IdMapIndex` (full f32 + scalar dot product in the
// POC) for vector indexing and ANN search.
//
// Out of POC scope: quantization, SIMD/GPU kernels, filtered search,
// reindex. The adapter wires the architecture so those can land later by
// replacing internals — see prompt.md.
//
// Lifecycle isolation: this file requires `IdMapIndex` from the dedicated
// sub-export `@qvac/embed-llamacpp/idMapIndex`, which is structured so it
// does NOT trigger fabric LLM/BERT backend initialization. Loading this
// adapter must not load any model.

const BaseDBAdapter = require('./BaseDBAdapter')
const { QvacErrorRAG, ERR_CODES } = require('../../errors')
const QvacLogger = require('@qvac/logging')
const HyperDB = require('hyperdb')
const dbSpec = require('./hyperspec/hyperdb/index.js')
const stringIdToU64 = require('../../utils/stringIdToU64')

const IdMapIndex = require('@qvac/embed-llamacpp/idMapIndex')

// u64 sentinel returned by IdMapIndex.search() when the index holds fewer
// than k entries. Matches fabric's `UINT64_MAX`.
const U64_SENTINEL = (1n << 64n) - 1n

// Reverse-map (bigint u64 -> string id) is built in-memory on adapter
// `open()` by scanning `@rag/documents` and pulling each row's
// `metadata.__turbovec_u64Id` field. Decision rationale (see prompt.md
// Phase 3): we can't extend the dbSpec without regenerating it; stashing
// the u64 hash inside the `metadata` JSON column is the smallest-blast
// alternative. Memory cost: 16 bytes per doc (vs. e.g. a sidecar table).
//
// `__turbovec_u64Id` is stored as a base-10 string so it survives JSON
// (`bigint` does not). Decoded back to BigInt on read.
const U64_META_KEY = '__turbovec_u64Id'

class TurboVecHybridAdapter extends BaseDBAdapter {
  /**
   * @param {Object} config
   * @param {Corestore} config.store - Corestore for HyperDB (documents)
   * @param {number} config.dim - vector dimensionality
   * @param {number} [config.bitWidth=4] - reserved (POC stores full f32)
   * @param {string} config.indexPath - filesystem path for the .tvim file
   * @param {string} config.embeddingModelId - propagated to documents rows
   * @param {string} [config.dbName='turbovec-store'] - hypercore name
   * @param {string} [config.documentsTable='@rag/documents']
   * @param {string} [config.configTable='@rag/config']
   * @param {QvacLogger} [config.logger]
   */
  constructor (config = {}) {
    super(config)
    if (!config.store) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: 'TurboVecHybridAdapter: `store` is required'
      })
    }
    if (typeof config.dim !== 'number' || config.dim <= 0) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: 'TurboVecHybridAdapter: `dim` must be a positive integer'
      })
    }
    if (typeof config.indexPath !== 'string' || config.indexPath.length === 0) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: 'TurboVecHybridAdapter: `indexPath` is required'
      })
    }
    if (typeof config.embeddingModelId !== 'string' ||
        config.embeddingModelId.length === 0) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: 'TurboVecHybridAdapter: `embeddingModelId` is required'
      })
    }
    this.store = config.store
    this.dim = config.dim
    this.bitWidth = config.bitWidth ?? 4
    this.indexPath = config.indexPath
    this.embeddingModelId = config.embeddingModelId
    this.dbName = config.dbName || 'turbovec-store'
    this.documentsTable = config.documentsTable || '@rag/documents'
    this.configTable = config.configTable || '@rag/config'
    this.logger = config.logger || new QvacLogger()

    this.hypercore = null
    this.db = null
    this.idx = null
    // bigint -> string id reverse map, populated on _open and maintained
    // through saveEmbeddings / deleteEmbeddings.
    this._u64ToStringId = new Map()
  }

  async _open () {
    this.logger.info('TurboVecHybridAdapter: opening')
    await this.store.ready()
    this.hypercore = this.store.get({ name: this.dbName })
    this.db = HyperDB.bee(this.hypercore, dbSpec, { autoUpdate: true })
    await this.db.ready()

    // Load or create the IdMapIndex. The .tvim file may not exist on
    // first open; in that case start fresh and let `write()` create it.
    const fs = await this._fs()
    if (this.indexPath && fs.existsSync(this.indexPath)) {
      this.logger.debug(
        `TurboVecHybridAdapter: loading index from ${this.indexPath}`)
      this.idx = await IdMapIndex.load(this.indexPath)
      if (this.idx.dim !== this.dim) {
        await this.idx.dispose()
        await this.db.close()
        throw new QvacErrorRAG({
          code: ERR_CODES.INVALID_PARAMS,
          adds: `TurboVecHybridAdapter: on-disk dim (${this.idx.dim}) ` +
            `does not match constructor dim (${this.dim})`
        })
      }
    } else {
      this.logger.debug('TurboVecHybridAdapter: creating fresh index')
      this.idx = new IdMapIndex({ dim: this.dim, bitWidth: this.bitWidth })
    }

    await this._rebuildReverseMap()
    this.isInitialized = true
    this.logger.info(
      `TurboVecHybridAdapter: open (n=${this.idx.length} vectors)`)
  }

  async _close () {
    this.logger.info('TurboVecHybridAdapter: closing')
    if (this.idx) {
      await this.idx.dispose()
      this.idx = null
    }
    if (this.db) {
      await this.db.close()
      this.db = null
    }
    this._u64ToStringId.clear()
    this.isInitialized = false
  }

  /**
   * Rebuilds the bigint -> string id reverse map by scanning the
   * documents table. Skips rows that lack the u64 metadata field (they
   * came from a different adapter or an older write).
   * @private
   */
  async _rebuildReverseMap () {
    this._u64ToStringId.clear()
    const snapshot = this.db.snapshot()
    try {
      const rows = await snapshot.find(this.documentsTable).toArray()
      for (const row of rows) {
        const raw = row?.metadata?.[U64_META_KEY]
        if (typeof raw === 'string' && raw.length > 0) {
          try {
            this._u64ToStringId.set(BigInt(raw), row.id)
          } catch (_) {
            // Malformed entry — skip; the IdMapIndex will surface its own
            // absence if a search returns it.
          }
        }
      }
    } finally {
      // hyperdb snapshots don't currently expose a close hook; GC handles it.
    }
  }

  async saveEmbeddings (embeddedDocs, opts = {}) {
    if (!Array.isArray(embeddedDocs) || embeddedDocs.length === 0) {
      return []
    }

    // Per-batch model-id sanity (mirrors HyperDBAdapter's contract).
    const modelIds = new Set(
      embeddedDocs.map((d) => d.embeddingModelId).filter(Boolean))
    if (modelIds.size > 1) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: 'TurboVecHybridAdapter: all docs in one batch must share embeddingModelId'
      })
    }
    if (modelIds.size === 1 &&
        !modelIds.has(this.embeddingModelId)) {
      this.logger.warn(
        `TurboVecHybridAdapter: doc embeddingModelId ${Array.from(modelIds)[0]} ` +
        `differs from adapter ${this.embeddingModelId}; using adapter value`)
    }

    const now = new Date()
    const n = embeddedDocs.length
    const vectors = new Float32Array(n * this.dim)
    const u64Ids = new BigUint64Array(n)
    const docMeta = new Array(n)

    for (let i = 0; i < n; i++) {
      const doc = embeddedDocs[i]
      if (!doc.embedding || doc.embedding.length !== this.dim) {
        throw new QvacErrorRAG({
          code: ERR_CODES.INVALID_PARAMS,
          adds: `TurboVecHybridAdapter: doc[${i}] (${doc.id}) embedding length ` +
            `${doc.embedding?.length} != adapter dim ${this.dim}`
        })
      }
      const u64 = stringIdToU64(doc.id)
      // Loud collision check: another doc in this adapter already maps
      // a different string id to the same u64.
      const prev = this._u64ToStringId.get(u64)
      if (prev !== undefined && prev !== doc.id) {
        throw new QvacErrorRAG({
          code: ERR_CODES.INVALID_PARAMS,
          adds: `TurboVecHybridAdapter: u64 hash collision between ` +
            `"${doc.id}" and "${prev}"`
        })
      }
      for (let j = 0; j < this.dim; j++) {
        vectors[i * this.dim + j] = doc.embedding[j]
      }
      u64Ids[i] = u64
      docMeta[i] = { id: doc.id, doc, u64 }
    }

    // Insert into the native index first. If addWithIds throws (duplicate
    // u64 across batches, dim mismatch, OOM), HyperDB stays untouched —
    // the operation is atomic from the caller's perspective.
    this.idx.addWithIds(vectors, u64Ids)

    // Persist documents to HyperDB and update the reverse map.
    const tx = await this.db.exclusiveTransaction()
    try {
      for (const { id, doc, u64 } of docMeta) {
        const metadata = { ...(doc.metadata || {}) }
        metadata[U64_META_KEY] = u64.toString(10)
        // Stash embeddingModelId in metadata too (the documents schema
        // doesn't have a dedicated column, but we want it on each row
        // for parity with HyperDBAdapter consumers).
        metadata.embeddingModelId = doc.embeddingModelId || this.embeddingModelId
        await tx.insert(this.documentsTable, {
          id,
          content: doc.content ?? '',
          contentHash: doc.contentHash ?? '',
          metadata,
          createdAt: now,
          updatedAt: now
        })
        this._u64ToStringId.set(u64, id)
      }
      await tx.flush()
    } catch (err) {
      // Roll back the native index inserts to keep HyperDB and IdMapIndex
      // in sync; the doc rows in this tx were never flushed.
      try { await tx.rollback() } catch (_) {}
      for (const { u64 } of docMeta) {
        try { this.idx.remove(u64) } catch (_) {}
        this._u64ToStringId.delete(u64)
      }
      throw err
    }

    // Persist the index after a successful batch. POC: full rewrite each
    // call — simple, correct, slow. Optimization phase replaces this with
    // either an append-only journal or a delta log.
    this.idx.write(this.indexPath)

    return docMeta.map(({ id }) => ({ id, status: 'fulfilled' }))
  }

  async search (query, queryVector, params = {}) {
    const topK = params.topK || 5
    if (!queryVector || queryVector.length !== this.dim) {
      throw new QvacErrorRAG({
        code: ERR_CODES.INVALID_PARAMS,
        adds: `TurboVecHybridAdapter.search: queryVector length ` +
          `${queryVector?.length} != dim ${this.dim}`
      })
    }
    const f32 = queryVector instanceof Float32Array
      ? queryVector
      : Float32Array.from(queryVector)

    const { scores, ids } = this.idx.search(f32, topK)

    const snapshot = this.db.snapshot()
    const results = []
    for (let i = 0; i < topK; i++) {
      const u64 = ids[i]
      if (u64 === U64_SENTINEL) continue
      const stringId = this._u64ToStringId.get(u64)
      if (!stringId) continue // stale reverse-map entry; skip silently
      const row = await snapshot.get(this.documentsTable, { id: stringId })
      if (!row) continue
      results.push({
        id: stringId,
        content: row.content,
        score: scores[i],
        metadata: row.metadata
      })
    }
    return results
  }

  async deleteEmbeddings (ids) {
    if (!Array.isArray(ids) || ids.length === 0) {
      throw new QvacErrorRAG({ code: ERR_CODES.INVALID_PARAMS })
    }
    const tx = await this.db.exclusiveTransaction()
    let removedAny = false
    try {
      for (const id of ids) {
        const u64 = stringIdToU64(id)
        const removed = this.idx.remove(u64)
        if (removed) removedAny = true
        this._u64ToStringId.delete(u64)
        await tx.delete(this.documentsTable, { id })
      }
      await tx.flush()
    } catch (err) {
      try { await tx.rollback() } catch (_) {}
      throw err
    }
    if (removedAny) {
      this.idx.write(this.indexPath)
    }
    return true
  }

  async reindex () {
    return {
      reindexed: false,
      details: { reason: 'data-oblivious; no rebuild needed' }
    }
  }

  // Lazy fs accessor — both Bare and Node ship a file API. Using
  // `bare-fs` directly works under Bare; on Node, `fs` is the equivalent.
  // The adapter only needs `existsSync` so the resolution surface stays
  // tiny.
  async _fs () {
    if (this._fsCache) return this._fsCache
    try {
      this._fsCache = require('bare-fs')
    } catch (_) {
      // eslint-disable-next-line node/no-missing-require
      this._fsCache = require('fs')
    }
    return this._fsCache
  }
}

module.exports = TurboVecHybridAdapter

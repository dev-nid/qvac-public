'use strict'

const { QvacErrorRAG, ERR_CODES } = require('./src/errors')

require('./src/types')

// RAG
const RAG = require('./src/RAG')

// Database Adapters
const BaseDBAdapter = require('./src/adapters/database/BaseDBAdapter')
const HyperDBAdapter = require('./src/adapters/database/HyperDBAdapter')
// TurboVecHybridAdapter is intentionally lazy-loaded below (it pulls in
// `@qvac/embed-llamacpp`'s native addon, so unconditional require would
// regress `HyperDBAdapter`-only consumers by forcing a native-binary
// dependency on `require('@qvac/rag')`). Consumers that want it should
// either access it via the lazy getter exposed on this module's exports,
// or import the dedicated sub-export `@qvac/rag/turbovec` to opt in
// explicitly.

// Chunker Adapters
const BaseChunkAdapter = require('./src/adapters/chunker/BaseChunkAdapter')
const LLMChunkAdapter = require('./src/adapters/chunker/LLMChunkAdapter')

// LLM Adapters
const BaseLlmAdapter = require('./src/adapters/llm/BaseLlmAdapter')
const QvacLlmAdapter = require('./src/adapters/llm/QvacLlmAdapter')
const HttpLlmAdapter = require('./src/adapters/llm/HttpLlmAdapter')

// Schemas
const embeddingSchemas = require('./src/schemas/embedding')

module.exports = {
  RAG,
  HyperDBAdapter,
  LLMChunkAdapter,
  BaseDBAdapter,
  BaseChunkAdapter,
  BaseLlmAdapter,
  HttpLlmAdapter,
  QvacLlmAdapter,
  QvacErrorRAG,
  ERR_CODES,
  ...embeddingSchemas
}

// Lazy getter: first access triggers the require, which loads
// `@qvac/embed-llamacpp` and its native binary. Consumers that never
// touch this property never pay the cost.
let _TurboVecHybridAdapter
Object.defineProperty(module.exports, 'TurboVecHybridAdapter', {
  enumerable: true,
  configurable: false,
  get () {
    if (_TurboVecHybridAdapter === undefined) {
      _TurboVecHybridAdapter =
        require('./src/adapters/database/TurboVecHybridAdapter')
    }
    return _TurboVecHybridAdapter
  }
})

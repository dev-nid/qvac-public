'use strict'
//
// Maps a string document id to an unsigned 64-bit integer for use as an
// external id in the fabric vector index (IdMapIndex). The mapping is the
// first 8 bytes of a SHA-256 of the UTF-8 string, decoded little-endian.
//
// Collision math: for a uniformly distributed 64-bit hash, the birthday
// probability of any collision over N documents is roughly N^2 / 2^65.
// At N = 100,000 the probability is ~5.4e-10. We treat collisions as a
// loud failure path: `IdMapIndex.addWithIds` throws on duplicate u64 ids,
// surfacing the collision rather than silently overwriting.

const qvacCrypto = require('#crypto')

/**
 * Deterministic, well-distributed string -> uint64 mapping.
 * @param {string} s - non-empty document id
 * @returns {bigint} unsigned 64-bit value
 */
function stringIdToU64 (s) {
  if (typeof s !== 'string' || s.length === 0) {
    throw new TypeError('stringIdToU64: id must be a non-empty string')
  }
  const digest = qvacCrypto.createHash('sha256').update(s).digest()
  // First 8 bytes, little-endian.
  let acc = 0n
  for (let i = 7; i >= 0; i--) {
    acc = (acc << 8n) | BigInt(digest[i])
  }
  return acc
}

module.exports = stringIdToU64

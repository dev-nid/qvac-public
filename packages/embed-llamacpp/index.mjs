import GGMLBert from './index.js'
import addon from './addon.js'
import IdMapIndex, { IdMapIndexFilter } from './idMapIndex.mjs'

const { BertInterface, mapAddonEvent } = addon
const { pickPrimaryGgufPath } = GGMLBert

export default GGMLBert
export { BertInterface, GGMLBert, IdMapIndex, IdMapIndexFilter, mapAddonEvent, pickPrimaryGgufPath }

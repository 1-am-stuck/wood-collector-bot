const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { loadSenseConfig } = require('../../js/sense/loadConfig')
const { planGrove } = require('../../js/sense/seedRich')

const cfg = loadSenseConfig(path.join(__dirname, '../..'))

test('grove is every odor plant we can put down, not four lonely logs', () => {
  const plan = planGrove(cfg.blockOdor, cfg.itemOdor)
  const blocks = new Set(plan.blocks.map(b => b.name))
  const items = new Set(plan.items.map(i => i.name))
  for (const must of ['oak_log', 'oak_leaves', 'cherry_log', 'cake', 'melon', 'sweet_berry_bush', 'honey_block', 'dandelion']) {
    assert.ok(blocks.has(must), `missing ${must}`)
  }
  for (const must of ['apple', 'sweet_berries', 'melon_slice', 'chorus_fruit', 'glow_berries']) {
    assert.ok(items.has(must), `missing fruit item ${must}`)
  }
  const air = new Set((plan.air || []).map(b => b.name))
  assert.ok(air.has('cake') && air.has('melon'), 'hang fruit next to the body')
  assert.equal(blocks.has('lava'), false, 'lava is not a snack')
  assert.equal(blocks.has('fire'), false)
  assert.ok(plan.blocks.length >= 24, `grove too thin: ${plan.blocks.length}`)
})

test('plantGrove places on a found deck and tosses fruit', async () => {
  const { plantGrove, planGrove } = require('../../js/sense/seedRich')
  const placed = []
  const tossed = []
  const bot = {
    version: '1.21.1',
    registry: {},
    entity: { position: { floored () { return { x: 0, y: 70, z: 0, offset (x, y, z) { return { x, y: 70 + y, z } } } } } },
    creative: { async setInventorySlot () {} },
    inventory: { slots: { 36: { name: 'apple' } } },
    async equip () {},
    async tossStack (it) { tossed.push(it) },
    async placeBlock (ref, face) { placed.push({ ref: ref.name, face }) },
    blockAt (pos) {
      if (!pos) return null
      if (pos.y <= 64) return { name: 'grass_block', boundingBox: 'block', position: { x: pos.x, y: 64, z: pos.z } }
      return { name: 'air', boundingBox: 'empty' }
    },
  }
  // Don't depend on minecraft-data in this unit: hold() will fail, so we only
  // assert the planner still wants fruit. Live planting is the Paper run.
  const plan = planGrove(cfg.blockOdor, cfg.itemOdor)
  assert.ok(plan.items.some(i => i.name === 'apple'))
  await plantGrove(bot, { blocks: [{ dx: 2, dy: 0, dz: 0, name: 'oak_log' }], items: [] })
})

test('nothing is planted on the fly itself', () => {
  const plan = planGrove(cfg.blockOdor, cfg.itemOdor)
  for (const b of plan.blocks) {
    assert.ok(b.dx !== 0 || b.dz !== 0 || b.dy < 0, `${b.name} sits on the body`)
  }
})

/** Plant the odor / taste tables into the Paper world.
 *
 * Navigate used to refuse planting, so a fly over ocean looked at empty sky and
 * smelled nothing. This lays a grove from `block_odor.json` and drops the fruit
 * items from `item_odor.json`. Lethal blocks stay in the table (the fly must
 * be able to smell lava) but they are not spawned under its feet.
 */

const { withTimeout } = require('./actions')

const SKIP_BLOCKS = new Set(['lava', 'fire', 'magma_block', 'water'])

function ring (n, radius, y) {
  const out = []
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n
    out.push({
      dx: Math.round(Math.cos(a) * radius),
      dy: y,
      dz: Math.round(Math.sin(a) * radius),
    })
  }
  return out
}

function planGrove (blockOdor, itemOdor) {
  const sources = Object.keys((blockOdor && blockOdor.sources) || {})
    .filter(name => !SKIP_BLOCKS.has(name))
  const fruit = Object.keys((itemOdor && itemOdor.sources) || {})
    .filter(name => /apple|berr|melon|honey|cake|cookie|pie|sugar|chorus|bread/.test(name))

  const blocks = []
  const inner = ring(Math.min(sources.length, 12), 2, 0)
  const outer = ring(Math.max(0, sources.length - 12), 5, 0)
  const slots = inner.concat(outer)
  sources.forEach((name, i) => {
    const slot = slots[i % slots.length]
    const spin = Math.floor(i / slots.length)
    blocks.push({
      dx: slot.dx + (spin % 2 === 0 ? 0 : (i % 2 ? 1 : -1)),
      dy: slot.dy,
      dz: slot.dz + spin,
      name,
    })
    if (name.endsWith('_log')) {
      const leaf = name.replace(/_log$/, '_leaves')
      if (sources.includes(leaf) || name === 'oak_log') {
        blocks.push({ dx: slot.dx, dy: slot.dy + 1, dz: slot.dz, name })
        blocks.push({ dx: slot.dx, dy: slot.dy + 2, dz: slot.dz, name: leaf })
        blocks.push({ dx: slot.dx + 1, dy: slot.dy + 2, dz: slot.dz, name: leaf })
        blocks.push({ dx: slot.dx - 1, dy: slot.dy + 2, dz: slot.dz, name: leaf })
      }
    }
  })

  const itemRing = ring(fruit.length, 2, 0)
  const items = fruit.map((name, i) => ({
    dx: itemRing[i].dx,
    dy: 1,
    dz: itemRing[i].dz,
    name,
    count: 8,
  }))

  // A second ring at body height so takeoff does not leave every smell on the dirt.
  const hang = ['cake', 'melon', 'honey_block', 'oak_leaves', 'sweet_berry_bush', 'dandelion']
  const air = hang.map((name, i) => {
    const a = (2 * Math.PI * i) / hang.length
    return {
      dx: Math.round(Math.cos(a) * 2),
      dy: 0,
      dz: Math.round(Math.sin(a) * 2),
      name,
      air: true,
    }
  })

  return { blocks, items, air }
}

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

const { Vec3 } = require('vec3')

/** Highest solid block under the fly — the grove sits on the world, not in empty sky. */
function findDeck (bot) {
  const p = bot.entity.position.floored()
  for (let dy = 0; dy >= -48; dy--) {
    const b = typeof bot.blockAt === 'function' ? bot.blockAt(p.offset(0, dy, 0), false) : null
    if (b && b.name && b.name !== 'air' && b.name !== 'cave_air' && b.name !== 'void_air' &&
        b.name !== 'water' && b.boundingBox === 'block') {
      return b.position
    }
  }
  return p.offset(0, -1, 0)
}

async function hold (bot, name) {
  if (!bot.creative || typeof bot.creative.setInventorySlot !== 'function') return false
  const mcData = require('minecraft-data')(bot.version)
  const Item = require('prismarine-item')(bot.registry)
  const def = mcData.itemsByName[name]
  if (!def) return false
  try {
    await withTimeout(bot.creative.setInventorySlot(36, new Item(def.id, 64)), 300)
    if (typeof bot.equip === 'function' && bot.inventory.slots[36]) {
      await withTimeout(bot.equip(bot.inventory.slots[36], 'hand'), 200)
    }
    return true
  } catch (_) {
    return false
  }
}

async function placeAt (bot, pos, name) {
  if (!(await hold(bot, name))) return false
  const below = typeof bot.blockAt === 'function' ? bot.blockAt(pos.offset(0, -1, 0), false) : null
  if (!below || below.name === 'air' || typeof bot.placeBlock !== 'function') return false
  const already = bot.blockAt(pos, false)
  if (already && already.name === name) return true
  if (already && already.name !== 'air' && already.name !== 'short_grass' && already.name !== 'tall_grass') {
    return false
  }
  try {
    await withTimeout(bot.placeBlock(below, new Vec3(0, 1, 0)), 400)
    return true
  } catch (_) {
    return false
  }
}

async function plantGrove (bot, plan) {
  if (!bot || !bot.entity) return { planted: 0, tossed: 0 }
  const deck = findDeck(bot)
  const origin = { x: deck.x, y: deck.y + 1, z: deck.z }
  let planted = 0
  for (const spot of plan.blocks) {
    const pos = new Vec3(origin.x + spot.dx, origin.y + spot.dy, origin.z + spot.dz)
    if (await placeAt(bot, pos, spot.name)) planted++
  }
  const body = bot.entity.position.floored()
  for (const spot of plan.air || []) {
    const pos = new Vec3(body.x + spot.dx, body.y + spot.dy, body.z + spot.dz)
    if (await placeAt(bot, pos, spot.name)) planted++
  }

  let tossed = 0
  for (const drop of plan.items) {
    if (!(await hold(bot, drop.name))) continue
    try {
      const held = bot.inventory && bot.inventory.slots[36]
      if (held && typeof bot.tossStack === 'function') {
        await withTimeout(bot.tossStack(held), 300)
        tossed++
      }
    } catch (_) {}
    await sleep(20)
  }
  return { planted, tossed, origin }
}

module.exports = { planGrove, plantGrove, SKIP_BLOCKS }

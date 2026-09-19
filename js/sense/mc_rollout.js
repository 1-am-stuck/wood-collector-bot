#!/usr/bin/env node
/** JSONL stdin/stdout Mineflayer env. Python train.py owns the GoalSpec pick. */

const fs = require('fs')
const path = require('path')
require('./loadEnv').loadRepoEnv()
const readline = require('readline')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const { loadSenseConfig } = require('./loadConfig')
const { sampleBot } = require('./senseBridge')
const { applyGoal } = require('./goalToSense')
const { applyAction, flyOffset, keepFlying, releaseControls, withTimeout } = require('./actions')
const { censusLogs, rarestLog, WOOD_LOGS } = require('./rarity')
const { compactFrame } = require('./logger')
const { stepReward } = require('./mc_reward')
const { navReward, newVisitSet } = require('./nav_reward')
const { sampleVoxels, samplePose, snapshotStale } = require('./voxelSnapshot')

const root = path.join(__dirname, '../..')
const cfg = loadSenseConfig(root)
const senseState = {}
const visited = []
let bot = null
let catalog = {}
let exploreRadius = 32
let lastFacts = {}
let lastMc = { host: '127.0.0.1', port: 25565, username: 'FruitFly' }
// Open-world exploring wants neither of these: no planted trees, no omniscient
// 64-block block search. Training turns them on.
let seedWoods = true
let censusEnabled = true
// 'rarest' scores progress toward the rarest log. 'navigate' scores covering ground
// without collisions and involves no goal, no census and no planted trees -- it is the
// stage that has to work before a GoalSpec goes back on top.
let mode = 'rarest'
let navCells = newVisitSet()
const view = { hz: 30, radiusXZ: 20, radiusY: 12, timer: null, voxels: null, voxelAt: 0 }

function send (obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

/**
 * Push pose at `view.hz` and the surrounding voxels only when they go stale.
 * The browser renders from these, so the camera costs no round trip and the
 * picture stays smooth even though the policy only decides once a second.
 */
function startStream (opts = {}) {
  stopStream()
  view.hz = Math.max(1, Math.min(60, opts.hz || view.hz))
  view.radiusXZ = opts.radiusXZ || view.radiusXZ
  view.radiusY = opts.radiusY || view.radiusY
  view.timer = setInterval(() => {
    if (!botAlive()) return
    try {
      const pose = samplePose(bot)
      if (pose) send({ stream: 'pose', t: Date.now(), pose })
      const now = Date.now()
      const stale = !view.voxels || snapshotStale(view.voxels, bot) || now - view.voxelAt > 4000
      if (stale) {
        const snap = sampleVoxels(bot, { radiusXZ: view.radiusXZ, radiusY: view.radiusY })
        if (snap) {
          view.voxels = snap
          view.voxelAt = now
          send({ stream: 'voxels', t: now, voxels: snap })
        }
      }
    } catch (_) {}
  }, Math.round(1000 / view.hz))
  if (view.timer.unref) view.timer.unref()
}

function stopStream () {
  if (view.timer) clearInterval(view.timer)
  view.timer = null
}

function inventoryCounts (bot) {
  const inv = {}
  for (const it of bot.inventory.items()) {
    inv[it.name] = (inv[it.name] || 0) + it.count
  }
  return inv
}

function scanLogs (bot) {
  const mcData = require('minecraft-data')(bot.version)
  const ids = WOOD_LOGS
    .map(n => mcData.blocksByName[n])
    .filter(Boolean)
    .map(b => b.id)
  if (ids.length === 0) return []
  const positions = bot.findBlocks({ matching: ids, maxDistance: 64, count: 80 })
  return positions.map(p => {
    const b = bot.blockAt(p)
    return { name: b ? b.name : 'oak_log', x: p.x, y: p.y, z: p.z }
  })
}

// findBlocks over a 64-block radius is far too slow to run at the policy tick,
// and the rarest-log census is an outer-loop fact anyway: it only changes as the
// bot explores. Refresh it about once a second and derive distance/bearing from
// the cached list every tick.
const CENSUS_TTL_MS = 1000
const censusCache = { at: 0, logs: [] }

function cachedLogs () {
  const now = Date.now()
  if (now - censusCache.at > CENSUS_TTL_MS) {
    censusCache.logs = scanLogs(bot)
    censusCache.at = now
  }
  return censusCache.logs
}

function rememberVisit (bot) {
  const p = bot.entity.position
  const x = Math.floor(p.x)
  const z = Math.floor(p.z)
  const last = visited[visited.length - 1]
  if (!last || last[0] !== x || last[1] !== z) visited.push([x, z])
}

function factsNow (goalSpec) {
  rememberVisit(bot)
  if (!censusEnabled) {
    return { census: {}, rarest: null, inventory: inventoryCounts(bot), distance: null, bearing: null, goal_id: null }
  }
  const logs = cachedLogs()
  const census = censusLogs(logs, visited, exploreRadius)
  const rarest = rarestLog(census)
  const inv = inventoryCounts(bot)
  let distance = null
  let bearing = null
  if (rarest) {
    const me = bot.entity.position
    let best = null
    let bestD = Infinity
    for (const b of logs) {
      if (b.name !== rarest) continue
      const d = Math.hypot(b.x - me.x, b.z - me.z)
      if (d < bestD) { bestD = d; best = b }
    }
    if (best) {
      distance = bestD
      const desired = Math.atan2(best.x - me.x, best.z - me.z)
      const err = Math.atan2(Math.sin(desired - bot.entity.yaw), Math.cos(desired - bot.entity.yaw))
      bearing = -err * 180 / Math.PI
    }
  }
  return {
    census,
    rarest,
    inventory: inv,
    distance,
    bearing,
    goal_id: goalSpec && goalSpec.id,
  }
}

/**
 * What the open-world navigation reward needs: where the fly is, and what it is
 * touching. Read off the frame the policy itself saw, so the reward cannot be scored
 * against contacts the fly was never told about.
 */
function navState (frame) {
  const p = bot.entity.position
  return {
    x: p.x, y: p.y, z: p.z,
    touchHead: !!frame.touchHead,
    touchWing: !!frame.touchWing,
    touchLegs: !!frame.touchLegs,
    touchNotum: !!frame.touchNotum,
    damage: frame.damage || 0,
    dead: bot.health != null && bot.health <= 0,
    onGround: !!frame.legsOnGround,
  }
}

function observe (goalSpec) {
  const worldFrame = sampleBot(bot, cfg, senseState)
  const world = {
    position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z },
    blocks: goalSpec ? cachedLogs() : [],
    visited,
    exploreRadius,
    census: lastFacts.census,
    rarest: lastFacts.rarest,
  }
  const frame = goalSpec ? applyGoal(worldFrame, goalSpec, world) : worldFrame
  lastFacts = factsNow(goalSpec)
  lastFacts.nav = navState(frame)
  return { frame: compactFrame(frame), facts: lastFacts }
}

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function seedWoodsIfNeeded () {
  const logs = scanLogs(bot)
  const types = new Set(logs.map(l => l.name))
  if (types.size >= 3) return
  const mcData = require('minecraft-data')(bot.version)
  const origin = bot.entity.position.floored()
  const spots = [
    [8, 0, 6, 'oak_log'],
    [14, 0, -8, 'birch_log'],
    [-10, 0, 12, 'spruce_log'],
    [4, 0, 18, 'cherry_log'],
  ]
  for (const [dx, dy, dz, name] of spots) {
    const block = mcData.blocksByName[name]
    if (!block) continue
    try {
      await withTimeout(bot.creative.setBlock(origin.offset(dx, dy, dz), block.id), 400)
    } catch (_) {}
  }
}

function botAlive () {
  return !!(bot && bot.entity && bot.game)
}

async function startFlying () {
  if (!bot || !bot.creative) return
  try { bot.creative.startFlying() } catch (_) {}
}

async function tossLogs () {
  if (!bot || !bot.inventory) return
  for (const it of bot.inventory.items()) {
    if (String(it.name).endsWith('_log')) {
      try { await withTimeout(bot.tossStack(it), 400) } catch (_) {}
    }
  }
}

async function ensureBot () {
  if (botAlive()) return
  bot = null
  await connect(lastMc)
  if (!botAlive()) throw new Error('minecraft bot not connected')
}

async function connect (opts) {
  lastMc = {
    host: (opts && opts.host) || lastMc.host,
    port: (opts && opts.port) || lastMc.port,
    username: (opts && opts.username) || lastMc.username,
  }
  if (botAlive()) return
  if (bot) {
    try { bot.end() } catch (_) {}
    bot = null
  }
  await new Promise((resolve, reject) => {
    bot = mineflayer.createBot({
      host: lastMc.host,
      port: lastMc.port,
      username: lastMc.username,
    })
    const t = setTimeout(() => reject(new Error('minecraft connect timeout')), 30000)
    bot.once('spawn', () => { clearTimeout(t); resolve() })
    bot.once('error', reject)
    bot.once('end', () => { bot = null })
  })
  await startFlying()
  if (!botAlive()) return
  // prismarine-viewer does not support 1.21.11; the dashboard renders the voxel
  // stream instead, so there is nothing to start here.
  view.voxels = null
  startStream(view)
  if (seedWoods) await seedWoodsIfNeeded()
}

async function handle (msg) {
  if (msg.cmd === 'hello') {
    if (msg.catalog_path) catalog = JSON.parse(JSON.stringify(require('yaml') ? {} : {}))
    return { ok: true }
  }
  if (msg.cmd === 'load_catalog') {
    catalog = msg.catalog
    exploreRadius = msg.explore_radius || 32
    return { ok: true, logs: Object.keys(catalog) }
  }
  if (msg.cmd === 'connect') {
    if (msg.view) {
      view.hz = msg.view.hz || view.hz
      view.radiusXZ = msg.view.radiusXZ || view.radiusXZ
      view.radiusY = msg.view.radiusY || view.radiusY
    }
    if (msg.seed_woods != null) seedWoods = !!msg.seed_woods
    if (msg.census != null) censusEnabled = !!msg.census
    if (msg.mode) mode = msg.mode
    // Navigating has no goal, so a census and planted trees would be doing nothing but
    // costing a 64-block block search every second.
    if (mode === 'navigate') {
      seedWoods = false
      censusEnabled = false
    }
    await connect(msg.minecraft || {})
    return {
      ok: true,
      username: bot && bot.username,
      view: { hz: view.hz, radiusXZ: view.radiusXZ, radiusY: view.radiusY },
      seed_woods: seedWoods,
      census: censusEnabled,
      mode,
    }
  }
  if (msg.cmd === 'play_reset') {
    await ensureBot()
    lastFacts = {}
    navCells = newVisitSet()
    // Stay airborne. This is a fly, not a pedestrian: landing it is what made
    // the body look dead, because DNp01 is takeoff and walking keys do nothing
    // with gravity off unless we translate the pose ourselves.
    keepFlying(bot)
    const obs = observe(null)
    if (mode === 'navigate') navReward(null, obs.facts.nav, navCells)
    return { ...obs, reward: null, done: false }
  }
  if (msg.cmd === 'observe') {
    await ensureBot()
    return { ...observe(null), reward: null, done: false }
  }
  if (msg.cmd === 'reset') {
    await ensureBot()
    visited.length = 0
    lastFacts = {}
    censusCache.at = 0
    // Novelty is per episode: a fresh set, so ground covered last episode is new again.
    navCells = newVisitSet()
    releaseControls(bot)
    await tossLogs()
    await startFlying()
    const ox = (Math.random() * 2 - 1) * 36
    const oz = (Math.random() * 2 - 1) * 36
    await flyOffset(bot, 0, 10, 0, 600)
    await flyOffset(bot, ox, 8, oz, 1200)
    keepFlying(bot)
    if (seedWoods) await seedWoodsIfNeeded()
    await sleep(80)
    rememberVisit(bot)
    const peek = factsNow(null)
    const spec = mode === 'navigate' || !peek.rarest ? null : catalog[peek.rarest]
    const obs = observe(spec)
    if (mode === 'navigate') {
      // Registers the starting cell without paying for it.
      navReward(null, obs.facts.nav, navCells)
    } else if (obs.facts.rarest) {
      bot.chat('rarest ' + obs.facts.rarest)
    }
    return { ...obs, reward: 0, done: false }
  }
  if (msg.cmd === 'step') {
    await ensureBot()
    const prev = {
      inventory: { ...(lastFacts.inventory || {}) },
      rarest: lastFacts.rarest,
      distance: lastFacts.distance,
    }
    const prevNav = lastFacts.nav
    await applyAction(bot, msg.action || 'noop', {
      ...cfg.actions.control,
      dtMs: cfg.actions.dtMs,
    })
    const spec = mode === 'navigate' || !lastFacts.rarest ? null : catalog[lastFacts.rarest]
    const obs = observe(spec)
    if (mode === 'navigate') {
      const scored = navReward(prevNav, obs.facts.nav, navCells)
      obs.facts.nav_cells = scored.cells
      obs.facts.nav_new_cell = scored.newCell
      obs.facts.nav_moved = scored.moved
      return { ...obs, reward: scored.reward, done: scored.done }
    }
    // No census means no goal, so there is no reward to report either.
    if (!censusEnabled) return { ...obs, reward: null, done: false }
    const scored = stepReward(prev, obs.facts)
    return { ...obs, reward: scored.reward, done: scored.done }
  }
  if (msg.cmd === 'close') {
    stopStream()
    if (bot) bot.end()
    return { ok: true }
  }
  throw new Error('unknown cmd ' + msg.cmd)
}

const rl = readline.createInterface({ input: process.stdin })
rl.on('line', async line => {
  if (!line.trim()) return
  try {
    const msg = JSON.parse(line)
    const res = await handle(msg)
    send(res)
  } catch (err) {
    send({ error: err.message })
  }
})

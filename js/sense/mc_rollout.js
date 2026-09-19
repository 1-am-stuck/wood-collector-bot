#!/usr/bin/env node
/** JSONL stdin/stdout Mineflayer env. Python train.py owns the GoalSpec pick. */

const fs = require('fs')
const path = require('path')
const readline = require('readline')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const { loadSenseConfig } = require('./loadConfig')
const { sampleBot } = require('./senseBridge')
const { applyGoal } = require('./goalToSense')
const { applyAction } = require('./actions')
const { censusLogs, rarestLog, WOOD_LOGS } = require('./rarity')
const { compactFrame } = require('./logger')

const root = path.join(__dirname, '../..')
const cfg = loadSenseConfig(root)
const senseState = {}
const visited = []
let bot = null
let catalog = {}
let exploreRadius = 32
let lastFacts = {}

function send (obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
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

function rememberVisit (bot) {
  const p = bot.entity.position
  const x = Math.floor(p.x)
  const z = Math.floor(p.z)
  const last = visited[visited.length - 1]
  if (!last || last[0] !== x || last[1] !== z) visited.push([x, z])
}

function factsNow (goalSpec) {
  const logs = scanLogs(bot)
  rememberVisit(bot)
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

function observe (goalSpec) {
  const worldFrame = sampleBot(bot, cfg, senseState)
  const logs = scanLogs(bot)
  const world = {
    position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z },
    blocks: logs,
    visited,
    exploreRadius,
    census: lastFacts.census,
    rarest: lastFacts.rarest,
  }
  const frame = goalSpec ? applyGoal(worldFrame, goalSpec, world) : worldFrame
  lastFacts = factsNow(goalSpec)
  return { frame: compactFrame(frame), facts: lastFacts }
}

async function connect (opts) {
  if (bot) return
  await new Promise((resolve, reject) => {
    bot = mineflayer.createBot({
      host: opts.host || '127.0.0.1',
      port: opts.port || 25565,
      username: opts.username || 'FruitFly',
    })
    const t = setTimeout(() => reject(new Error('minecraft connect timeout')), 30000)
    bot.once('spawn', () => { clearTimeout(t); resolve() })
    bot.once('error', reject)
    bot.once('end', () => { bot = null })
  })
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
    await connect(msg.minecraft || {})
    return { ok: true, username: bot.username }
  }
  if (msg.cmd === 'reset') {
    visited.length = 0
    lastFacts = {}
    rememberVisit(bot)
    lastFacts = factsNow(null)
    const spec = lastFacts.rarest ? catalog[lastFacts.rarest] : null
    const obs = observe(spec)
    return { ...obs, reward: 0, done: false }
  }
  if (msg.cmd === 'step') {
    const prev = { ...lastFacts }
    const specBefore = lastFacts.rarest ? catalog[lastFacts.rarest] : null
    await applyAction(bot, msg.action || 'noop', cfg.actions.control)
    lastFacts = factsNow(specBefore)
    const spec = lastFacts.rarest ? catalog[lastFacts.rarest] : specBefore
    const obs = observe(spec)
    let reward = 0
    if (spec && spec.success && spec.success.type === 'inventory_contains') {
      const item = spec.success.item
      const a = (obs.facts.inventory || {})[item] || 0
      const b = (prev.inventory || {})[item] || 0
      if (a > b) reward += 15
      if (obs.facts.distance != null && prev.distance != null) {
        reward += (prev.distance - obs.facts.distance) * 0.3
      }
    } else if (obs.facts.rarest) {
      reward += 0.05
    }
    const done = reward >= 15
    return { ...obs, reward, done }
  }
  if (msg.cmd === 'close') {
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

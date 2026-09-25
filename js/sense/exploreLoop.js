/**
 * Live fly: SenseBridge every tick, walk/turn/jump/mine in Minecraft.
 * GoalToSense stays generic — we only swap which GoalSpec applyGoal sees.
 */
const { sampleBot } = require('./senseBridge')
const { applyGoal } = require('./goalToSense')
const { applyAction } = require('./actions')
const { censusLogs, rarestLog, WOOD_LOGS } = require('./rarity')
const { compactFrame, openJsonl } = require('./logger')

function specForLog (name, odorSources) {
  const pattern = odorSources[name]
  if (!pattern) return null
  return {
    id: `collect:${name}`,
    injections: [
      { modality: 'olfaction', pattern, strength: 0.15, falloff: 'tonic' },
      {
        modality: 'vision',
        population: 'LC11',
        strength: 0.4,
        falloff: 'when_seen',
        source_query: { kind: 'block', name },
      },
    ],
    success: { type: 'inventory_contains', item: name, count: 1 },
  }
}

function scanLogs (bot) {
  const mcData = require('minecraft-data')(bot.version)
  const ids = WOOD_LOGS.map(n => mcData.blocksByName[n]).filter(Boolean).map(b => b.id)
  if (!ids.length) return []
  return bot.findBlocks({ matching: ids, maxDistance: 64, count: 80 }).map(p => {
    const b = bot.blockAt(p)
    return { name: b ? b.name : 'oak_log', x: p.x, y: p.y, z: p.z }
  })
}

function pickReactiveAction (frame, facts) {
  const loom = (frame.objectChannels && frame.objectChannels.LC4) || 0
  if (loom > 0.45) return 'turn_left'
  if (frame.touchHead > 0.5 || frame.touchLegs > 0.5) return 'jump'
  if (facts.rarest && facts.distance != null && facts.distance < 3 && Math.abs(facts.bearing || 0) < 40) {
    return 'mine'
  }
  if (facts.bearing != null) {
    if (facts.bearing > 12) return 'turn_right'
    if (facts.bearing < -12) return 'turn_left'
    return 'forward'
  }
  if (frame.odorBearingDeg != null) {
    if (frame.odorBearingDeg > 15) return 'turn_right'
    if (frame.odorBearingDeg < -15) return 'turn_left'
    return 'forward'
  }
  if (Math.random() < 0.12) return Math.random() < 0.5 ? 'turn_left' : 'turn_right'
  if (Math.random() < 0.06) return 'jump'
  return 'forward'
}

function startExplore (bot, cfg, opts = {}) {
  const state = {}
  const visited = []
  const exploreRadius = opts.exploreRadius || 32
  const odorSources = cfg.blockOdor.sources
  let running = false
  let log = null
  let lastChat = 0

  function visit () {
    const p = bot.entity.position
    const x = Math.floor(p.x)
    const z = Math.floor(p.z)
    const last = visited[visited.length - 1]
    if (!last || last[0] !== x || last[1] !== z) visited.push([x, z])
  }

  function censusAndSpec () {
    visit()
    const logs = scanLogs(bot)
    const census = censusLogs(logs, visited, exploreRadius)
    const rarest = rarestLog(census)
    const spec = rarest ? specForLog(rarest, odorSources) : null
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
    return { logs, census, rarest, spec, distance, bearing }
  }

  async function tick () {
    const extra = censusAndSpec()
    const worldFrame = sampleBot(bot, cfg, state)
    const world = {
      position: {
        x: bot.entity.position.x,
        y: bot.entity.position.y,
        z: bot.entity.position.z,
      },
      blocks: extra.logs,
      visited,
      exploreRadius,
    }
    const frame = extra.spec ? applyGoal(worldFrame, extra.spec, world) : worldFrame
    const facts = {
      census: extra.census,
      rarest: extra.rarest,
      distance: extra.distance,
      bearing: extra.bearing,
      inventory: Object.fromEntries(
        bot.inventory.items().map(it => [it.name, it.count]),
      ),
    }
    const action = pickReactiveAction(frame, facts)
    if (log) {
      log.write({
        t: Date.now(),
        action,
        goal_id: extra.spec && extra.spec.id,
        facts,
        frame: compactFrame(frame),
        source: 'explore',
      })
    }
    const now = Date.now()
    if (now - lastChat > 8000) {
      lastChat = now
      const n = Object.keys(extra.census).length
      bot.chat(extra.rarest
        ? `rarest ${extra.rarest} (${extra.census[extra.rarest]}) · ${n} types seen`
        : 'exploring — no logs in census yet')
    }
    await applyAction(bot, action, cfg.actions.control)
    return { frame, facts, action }
  }

  async function run () {
    if (running) return
    running = true
    bot.chat('fly exploring')
    try {
      while (running) {
        if (!bot.entity) break
        await tick()
      }
    } finally {
      running = false
      bot.clearControlStates()
    }
  }

  function stop () {
    running = false
  }

  function openLog (file) {
    log = openJsonl(file)
    return log.path
  }

  return { run, stop, tick, openLog, isRunning: () => running }
}

module.exports = { startExplore, specForLog, pickReactiveAction }

const { Vec3 } = require('vec3')

const WOOD_TYPES = [
  'oak', 'spruce', 'birch', 'jungle', 'acacia',
  'dark_oak', 'mangrove', 'cherry', 'pale_oak',
]

const TRAIL_MAX = 20
const HOP_MIN = 12
const HOP_MAX = 20

function missingWoodTypes (itemNames) {
  return WOOD_TYPES.filter(type => !itemNames.includes(`${type}_log`))
}

function nextAction ({ missing, nearbyTarget, targetDist }) {
  if (missing.length === 0) return { type: 'done' }
  if (nearbyTarget && targetDist != null && targetDist > 20) {
    return { type: 'approach', target: nearbyTarget }
  }
  if (nearbyTarget) return { type: 'collect', target: nearbyTarget }
  return { type: 'explore' }
}

function posKey (pos) {
  return `${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}`
}

// Flipping between two lengths (7↔8) is a wedge. A single repeating
// length (2,2,2…) is often a short real path — that is not stuck.
function isOscillating (recentPathLens) {
  return recentPathLens.length >= 6 && new Set(recentPathLens).size === 2
}

function nearestUnblacklisted (positions, blacklist, origin) {
  let best = null
  let bestDist = Infinity
  for (const p of positions) {
    if (blacklist.has(posKey(p))) continue
    const dist = p.distanceTo(origin)
    if (dist < bestDist) {
      best = p
      bestDist = dist
    }
  }
  return best
}

function blacklistTree (blacklist, pos) {
  for (let dx = -3; dx <= 3; dx++) {
    for (let dy = -6; dy <= 6; dy++) {
      for (let dz = -3; dz <= 3; dz++) {
        blacklist.add(posKey({ x: pos.x + dx, y: pos.y + dy, z: pos.z + dz }))
      }
    }
  }
}

function rememberHere (trail, p) {
  const last = trail[trail.length - 1]
  if (last && last.equals(p)) return
  trail.push(p)
  if (trail.length > TRAIL_MAX) trail.shift()
}

function isStandable (bot, x, y, z) {
  const feet = bot.blockAt(new Vec3(x, y, z))
  const head = bot.blockAt(new Vec3(x, y + 1, z))
  const floor = bot.blockAt(new Vec3(x, y - 1, z))
  if (!feet || !head || !floor) return false
  return feet.boundingBox === 'empty' && head.boundingBox === 'empty' && floor.boundingBox === 'block'
}

function surfaceAt (bot, x, z, fromY) {
  const y0 = Math.floor(fromY)
  for (let y = y0 + 6; y >= y0 - 10; y--) {
    if (isStandable(bot, x, y, z)) return new Vec3(x, y, z)
  }
  return null
}

function scoreHop (here, dest, toward, recentKeys, scan) {
  const dy = Math.abs(dest.y - here.y)
  if (dy > 6) return -Infinity
  if (recentKeys && recentKeys.has(posKey(dest))) return -Infinity
  let score = 0
  if (toward) {
    const before = Math.hypot(toward.x - here.x, toward.z - here.z)
    const after = Math.hypot(toward.x - dest.x, toward.z - dest.z)
    score = before - after - dy * 2
  } else {
    score = Math.hypot(dest.x - here.x, dest.z - here.z) - dy * 4
  }
  const clear = clearanceToward(scan, here, dest)
  const hopDist = Math.hypot(dest.x - here.x, dest.z - here.z)
  if (clear < 2.5) score -= 25
  else if (clear < hopDist * 0.45) score -= 12
  else score += Math.min(clear, 10) * 0.4
  return score
}

function clearanceToward (scan, origin, dest) {
  if (!scan || scan.length === 0) return 8
  const yaw = Math.atan2(-(dest.x - origin.x), -(dest.z - origin.z))
  let best = scan[0]
  let bestDiff = Infinity
  for (const reading of scan) {
    const diff = Math.abs(wrapAngle(reading.yaw - yaw))
    if (diff < bestDiff) {
      bestDiff = diff
      best = reading
    }
  }
  return best.clearance
}

function isEmpty (block) {
  return !!block && block.boundingBox === 'empty'
}

function isSolid (block) {
  return !!block && block.boundingBox === 'block'
}

function mapLocal (bot) {
  const p = bot.entity.position.floored()
  const dirs = [
    { dx: 1, dz: 0, name: 'E' },
    { dx: -1, dz: 0, name: 'W' },
    { dx: 0, dz: 1, name: 'S' },
    { dx: 0, dz: -1, name: 'N' },
  ]
  const cells = []
  const exits = []
  for (const dir of dirs) {
    const atFeet = bot.blockAt(p.offset(dir.dx, 0, dir.dz))
    const atHead = bot.blockAt(p.offset(dir.dx, 1, dir.dz))
    const above = bot.blockAt(p.offset(dir.dx, 2, dir.dz))
    const walkable = isEmpty(atFeet) && isEmpty(atHead)
    // 1-block pit: solid rim at feet, air to land on top of it.
    const stepUp = isSolid(atFeet) && isEmpty(atHead) && isEmpty(above)
    cells.push({
      dx: dir.dx,
      dz: dir.dz,
      name: dir.name,
      walkable,
      stepUp,
      solid: isSolid(atFeet),
      feet: atFeet ? atFeet.name : '?',
      head: atHead ? atHead.name : '?',
    })
    if (stepUp) exits.push({ dx: dir.dx, dz: dir.dz, name: dir.name, feet: atFeet.name })
  }
  const boxedAtFeet = cells.filter(c => c.solid).length >= 3 && !cells.some(c => c.walkable)
  if (boxedAtFeet && exits.length === 0) {
    exits.push({ dx: 1, dz: 0, name: 'E' })
  }
  exits.sort((a, b) => {
    const logA = a.feet && a.feet.endsWith('_log') ? 1 : 0
    const logB = b.feet && b.feet.endsWith('_log') ? 1 : 0
    return logA - logB
  })
  return {
    cells,
    exits,
    inHole: boxedAtFeet,
    holeExit: exits[0] || null,
  }
}

function bestStepUp (map, from, toward, opts = {}) {
  const mustFace = opts.mustFace !== false
  const steps = map.cells.filter(c => c.stepUp)
  if (steps.length === 0) return null
  if (!toward) return mustFace ? null : steps[0]
  const tx = toward.x - from.x
  const tz = toward.z - from.z
  let best = null
  let bestDot = -Infinity
  for (const step of steps) {
    const dot = step.dx * tx + step.dz * tz
    if (dot > bestDot) {
      bestDot = dot
      best = step
    }
  }
  if (mustFace && bestDot <= 0) return null
  return best
}

function wrapAngle (a) {
  while (a > Math.PI) a -= Math.PI * 2
  while (a < -Math.PI) a += Math.PI * 2
  return a
}

function pickExploreGoal (bot, here, toward, recentKeys, scan) {
  const origin = here.floored ? here.floored() : here
  const samples = []
  const baseAngle = toward
    ? Math.atan2(toward.z - origin.z, toward.x - origin.x)
    : Math.random() * Math.PI * 2

  for (let i = 0; i < 8; i++) {
    const angle = toward
      ? baseAngle + (i - 3.5) * 0.35
      : (i / 8) * Math.PI * 2 + Math.random() * 0.4
    const dist = HOP_MIN + Math.random() * (HOP_MAX - HOP_MIN)
    const x = Math.floor(origin.x + Math.cos(angle) * dist)
    const z = Math.floor(origin.z + Math.sin(angle) * dist)
    const dest = surfaceAt(bot, x, z, origin.y)
    if (!dest) continue
    const score = scoreHop(origin, dest, toward, recentKeys, scan)
    if (score === -Infinity) continue
    samples.push({ dest, score })
  }

  samples.sort((a, b) => b.score - a.score)
  return samples[0] ? samples[0].dest : null
}

module.exports = {
  WOOD_TYPES,
  missingWoodTypes,
  nextAction,
  isOscillating,
  nearestUnblacklisted,
  blacklistTree,
  rememberHere,
  pickExploreGoal,
  posKey,
  mapLocal,
  bestStepUp,
}

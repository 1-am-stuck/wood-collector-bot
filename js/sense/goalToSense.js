const { addOdor, addTaste } = require('./frame')
const { stripNs } = require('./senseBridge')

function cloneFrame (frame) {
  return {
    ...frame,
    luminance: frame.luminance.slice(),
    objects: frame.objects.map(o => ({ ...o })),
    odor: { ...frame.odor },
    taste: { ...frame.taste },
    objectChannels: { ...(frame.objectChannels || {}) },
  }
}

function nearestMatch (world, query) {
  if (!query || !world) return null
  const name = query.name
  const kind = query.kind
  const pos = world.position || { x: 0, y: 64, z: 0 }
  let best = null
  let bestD = Infinity
  const list = kind === 'entity' ? (world.entities || []) : kind === 'item' ? (world.items || []) : (world.blocks || [])
  for (const src of list) {
    const n = stripNs(src.name || src.type)
    if (n !== name && !(kind === 'entity' && (src.type === name))) continue
    const d = Math.hypot((src.x - pos.x), (src.y - pos.y), (src.z - pos.z))
    if (d < bestD) {
      bestD = d
      best = { ...src, distance: d, name: n }
    }
  }
  return best
}

function injectionWeight (inj, world) {
  const fall = inj.falloff || 'tonic'
  if (fall === 'tonic') return inj.strength
  const hit = nearestMatch(world, inj.source_query)
  if (!hit) return 0
  if (fall === 'when_seen') return inj.strength
  if (fall === 'when_near') {
    const d = hit.distance
    return inj.strength * (1 / (1 + d))
  }
  if (fall === 'distance') {
    return inj.strength * Math.exp(-hit.distance / (inj.lambda || 6))
  }
  return inj.strength
}

/**
 * Compile a GoalSpec into additive drive on a world-built SensoryFrame.
 * Does not replace world senses — only adds.
 */
function applyGoal (frame, goal, world = null) {
  const out = cloneFrame(frame)
  if (!goal || !goal.injections) return out
  for (const inj of goal.injections) {
    const w = injectionWeight(inj, world)
    if (w <= 0) continue
    if (inj.modality === 'olfaction' && inj.pattern) {
      for (const [glom, aff] of Object.entries(inj.pattern)) addOdor(out, glom, aff * w)
    } else if (inj.modality === 'gustation' && inj.pattern) {
      for (const [grn, aff] of Object.entries(inj.pattern)) addTaste(out, grn, aff * w)
    } else if (inj.modality === 'vision') {
      const pop = inj.population || 'LC11'
      out.objectChannels[pop] = Math.min(1.5, (out.objectChannels[pop] || 0) + w)
    } else if (inj.modality === 'mechano') {
      const field = inj.field || 'groomDust'
      out[field] = Math.min(1, (out[field] || 0) + w)
    }
  }
  return out
}

function goalSuccess (goal, facts) {
  if (!goal || !goal.success) return false
  const s = goal.success
  if (s.type === 'inventory_contains') {
    const n = (facts.inventory || {})[s.item] || 0
    return n >= (s.count || 1)
  }
  if (s.type === 'taste_contact') {
    return (facts.taste || {})[s.grn] >= (s.min || 0.5)
  }
  if (s.type === 'distance_above') {
    const d = facts.distances && facts.distances[s.entity]
    return d == null || d >= s.blocks
  }
  return false
}

function rewardFor (goal, prevFacts, facts) {
  if (!goal || !goal.success) return 0
  const s = goal.success
  let r = 0
  if (goalSuccess(goal, facts) && !goalSuccess(goal, prevFacts || {})) r += 10
  if (s.type === 'inventory_contains') {
    const a = (facts.inventory || {})[s.item] || 0
    const b = (prevFacts && prevFacts.inventory ? prevFacts.inventory[s.item] : 0) || 0
    r += (a - b) * 5
  }
  if (s.type === 'distance_above' && facts.distances && prevFacts && prevFacts.distances) {
    const d0 = prevFacts.distances[s.entity]
    const d1 = facts.distances[s.entity]
    if (d0 != null && d1 != null) r += (d1 - d0) * 0.1
  }
  if (facts.closerToQuery != null) r += facts.closerToQuery
  return r
}

module.exports = { applyGoal, cloneFrame, nearestMatch, injectionWeight, goalSuccess, rewardFor }

/**
 * Reward for learning to move, with no goal at all.
 *
 * "Navigates by itself" has to come before "collects wood". There is no GoalSpec here,
 * no rarest-log census, no target block and nothing injected into olfaction: the only
 * thing being learned is how to walk around a world without getting stuck.
 *
 * Three terms, and a reason for each:
 *
 * - **New ground.** The bulk of the reward is for entering a block cell the fly has
 *   not visited. Rewarding raw displacement instead would be fully paid out by
 *   spinning on the spot or pacing two blocks back and forth, which is exactly the
 *   failure this is meant to train out of the policy. Novelty only pays once per cell.
 * - **Contact costs.** Bumping the head, legs, wing or notum into geometry is a small
 *   penalty, and taking damage is a larger one. A fly that walks into a wall and keeps
 *   pushing is not navigating. Standing on the ground is not contact.
 * - **Stalling costs.** A small constant penalty whenever the fly barely moved, so
 *   doing nothing is never the safe option. It has to stay well under the new-cell
 *   reward or the policy learns that walking is not worth the risk of a bump.
 *
 * Deliberately absent: any bonus for facing a particular way, any shaping toward a
 * destination, and any override of the chosen action. The policy picks every step.
 */

// One reward unit for new ground; everything else is scaled against it so the balance
// is readable rather than a pile of magic numbers.
const NEW_CELL = 1.0
const MOVE_SCALE = 0.05      // per block of displacement, for gradient between cells
const CONTACT = -0.05        // per contact field, per tick
// Strictly more than NEW_CELL, so pushing into lava or a mob is never worth the ground
// it uncovers. At equal magnitude the fly is indifferent to being hurt while exploring,
// which is not a trade-off worth offering it.
const DAMAGE = -2.0
const STALL = -0.02
const STALL_BLOCKS = 0.05    // moved less than this in a tick counts as stalled
const FALL_DEATH = -5.0

/** Block cell a position falls in. Vertical is included so climbing counts. */
function cellKey (pos) {
  return `${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}`
}

/**
 * Tracks which cells have been visited across an episode.
 *
 * Kept as an explicit object rather than module state so a reset really does forget,
 * and so tests can drive it without a Minecraft server.
 */
function newVisitSet () {
  return new Set()
}

function contactCount (nav) {
  if (!nav) return 0
  let n = 0
  for (const key of ['touchHead', 'touchWing', 'touchLegs', 'touchNotum']) {
    if (nav[key]) n += 1
  }
  return n
}

/**
 * Score one tick of open-world movement.
 *
 * `prev` and `next` are `{ x, y, z, touchHead, ..., damage, dead }`. `visited` is a Set
 * from `newVisitSet()` and is mutated, because novelty is a property of the episode.
 */
function navReward (prev, next, visited) {
  if (!next) return { reward: 0, done: false, newCell: false, moved: 0 }
  if (!prev) {
    // First observation of an episode: register the starting cell without paying for
    // it, so spawning somewhere new is not itself a reward.
    if (visited) visited.add(cellKey(next))
    return { reward: 0, done: false, newCell: false, moved: 0 }
  }

  const moved = Math.hypot(next.x - prev.x, next.y - prev.y, next.z - prev.z)
  let reward = 0

  const key = cellKey(next)
  const fresh = visited ? !visited.has(key) : false
  if (visited) visited.add(key)
  if (fresh) reward += NEW_CELL
  reward += MOVE_SCALE * moved

  reward += CONTACT * contactCount(next)
  if (next.damage) reward += DAMAGE * Math.min(1, Number(next.damage) || 1)
  if (moved < STALL_BLOCKS) reward += STALL

  let done = false
  if (next.dead) {
    reward += FALL_DEATH
    done = true
  }

  return { reward, done, newCell: fresh, moved, cells: visited ? visited.size : 0 }
}

module.exports = {
  navReward,
  newVisitSet,
  cellKey,
  weights: { NEW_CELL, MOVE_SCALE, CONTACT, DAMAGE, STALL, STALL_BLOCKS, FALL_DEATH },
}

/** Reward is only for the current rarest log. GoalToSense is not involved. */

function count (inv, name) {
  return (inv && name) ? (inv[name] || 0) : 0
}

function stepReward (prev, facts) {
  const rarest = facts && facts.rarest
  if (!rarest) return { reward: 0, done: false }
  const gained = count(facts.inventory, rarest) - count(prev && prev.inventory, rarest)
  let reward = 0
  let done = false
  if (gained > 0) {
    reward += 15
    done = true
  }
  if (facts.distance != null && prev && prev.distance != null) {
    reward += (prev.distance - facts.distance) * 0.3
  }
  return { reward, done }
}

module.exports = { stepReward }

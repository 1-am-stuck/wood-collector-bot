/**
 * One number for feeding taxis: cover ground, then eat.
 *
 * Navigate still has to pay (there is no navigate checkpoint yet, and a fly that
 * never moves never contacts sugar). Taste is the real event: `rewardFor` on the
 * feed GoalSpec is +10 the first time labellar LB3b crosses 0.5. Sugar-odor
 * (DM1 / VA2) may rise a little toward fruit, but it is capped so following a
 * smell cannot outscore actually tasting.
 *
 * Deliberately absent: inventory bonuses. Picking up an apple is not feeding.
 */

const { navReward } = require('./nav_reward')
const { rewardFor, goalSuccess } = require('./goalToSense')

const SUGAR_GLOMS = ['DM1', 'VA2']
// Must stay well under the +10 taste bonus. The test asks that eating beats
// walking-plus-shaping by more than 8, so this cap is 0.2.
const ODOR_SHAPING_CAP = 0.2

function sugarOdor (frame) {
  const odor = (frame && frame.odor) || {}
  let s = 0
  for (const g of SUGAR_GLOMS) s += Number(odor[g]) || 0
  return s
}

/**
 * Score one tick of feed-mode walking.
 *
 * `prevNav` / `nextNav` are the same pose+contact objects `navReward` takes.
 * `facts.taste` must already be a copy of the frame's GRNs (the env copies it).
 */
function scoreFeedStep ({
  prevNav,
  nextNav,
  navCells,
  prevFacts,
  facts,
  frame,
  prevFrame,
  goal,
}) {
  const nav = navReward(prevNav, nextNav, navCells)
  const eat = rewardFor(goal, prevFacts || {}, facts || {})
  const shaping = Math.min(
    ODOR_SHAPING_CAP,
    Math.max(0, sugarOdor(frame) - sugarOdor(prevFrame)),
  )
  const done = !!(nav.done || goalSuccess(goal, facts || {}))
  return {
    reward: nav.reward + eat + shaping,
    done,
    nav,
    eat,
    shaping,
  }
}

module.exports = { scoreFeedStep, sugarOdor, ODOR_SHAPING_CAP }

const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { loadGoalSpec } = require('../../js/sense/loadConfig')
const { goalSuccess, rewardFor } = require('../../js/sense/goalToSense')
const { navReward, newVisitSet } = require('../../js/sense/nav_reward')
const { scoreFeedStep } = require('../../js/sense/feed_reward')

const root = path.join(__dirname, '../..')
const feed = loadGoalSpec('feed', root)

const at = (x, y, z, extra = {}) => ({ x, y, z, ...extra })

test('feed success is labellar sugar on facts.taste, not inventory', () => {
  assert.equal(goalSuccess(feed, { inventory: { cake: 8 } }), false)
  assert.equal(goalSuccess(feed, { taste: { LB3b: 0.05 } }), false)
  assert.equal(goalSuccess(feed, { taste: { LB3b: 0.5 } }), true)
  const r = rewardFor(feed, { taste: {} }, { taste: { LB3b: 0.8 } })
  assert.ok(r >= 10)
})

test('feed tick pays for new ground and for tasting sugar', () => {
  const cells = newVisitSet()
  navReward(null, at(0, 64, 0), cells)
  const walk = scoreFeedStep({
    prevNav: at(0, 64, 0),
    nextNav: at(0, 64, 1.2),
    navCells: cells,
    prevFacts: { taste: {} },
    facts: { taste: {} },
    frame: { odor: { DM1: 0.1, VA2: 0.1 } },
    prevFrame: { odor: { DM1: 0, VA2: 0 } },
    goal: feed,
  })
  assert.equal(walk.done, false)
  assert.ok(walk.reward > 0, 'covering ground should pay')

  const eat = scoreFeedStep({
    prevNav: at(0, 64, 1.2),
    nextNav: at(0, 64, 1.3),
    navCells: cells,
    prevFacts: { taste: {} },
    facts: { taste: { LB3b: 0.9 } },
    frame: { odor: { DM1: 0.4, VA2: 0.4 } },
    prevFrame: { odor: { DM1: 0.1, VA2: 0.1 } },
    goal: feed,
  })
  assert.equal(eat.done, true)
  assert.ok(eat.reward > walk.reward + 8, 'taste must dominate sugar-odor shaping')
})

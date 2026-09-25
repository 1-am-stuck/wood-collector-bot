const { test } = require('node:test')
const assert = require('node:assert/strict')
const { stepReward } = require('../../js/sense/mc_reward')

test('only the rarest log pays the collect bonus', () => {
  const prev = { inventory: { oak_log: 0, cherry_log: 0 }, rarest: 'cherry_log', distance: 8 }
  const oak = stepReward(prev, { inventory: { oak_log: 1, cherry_log: 0 }, rarest: 'cherry_log', distance: 8 })
  assert.equal(oak.reward, 0)
  assert.equal(oak.done, false)

  const cherry = stepReward(prev, { inventory: { oak_log: 0, cherry_log: 1 }, rarest: 'cherry_log', distance: 8 })
  assert.equal(cherry.reward, 15)
  assert.equal(cherry.done, true)
})

test('closing distance to the rarest log is shaped, no rarest is zero', () => {
  const closer = stepReward(
    { inventory: {}, rarest: 'oak_log', distance: 12 },
    { inventory: {}, rarest: 'oak_log', distance: 9 },
  )
  assert.ok(closer.reward > 0)
  assert.equal(closer.done, false)
  const none = stepReward({ inventory: {}, rarest: null, distance: null }, { inventory: {}, rarest: null, distance: null })
  assert.equal(none.reward, 0)
})

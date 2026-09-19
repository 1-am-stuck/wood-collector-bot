const { test } = require('node:test')
const assert = require('node:assert/strict')
const { pickReactiveAction, specForLog } = require('../../js/sense/exploreLoop')

test('specForLog is a normal GoalSpec, not a GoalToSense rewrite', () => {
  const spec = specForLog('cherry_log', { cherry_log: { VL2a: 0.7, D: 0.5 } })
  assert.equal(spec.id, 'collect:cherry_log')
  assert.equal(spec.injections[0].modality, 'olfaction')
  assert.equal(spec.success.item, 'cherry_log')
})

test('reactive fly walks toward a bearing and mines when close', () => {
  const frame = { objectChannels: {}, touchHead: 0, touchLegs: 0, odorBearingDeg: null }
  assert.equal(pickReactiveAction(frame, { rarest: 'oak_log', distance: 2, bearing: 5 }), 'mine')
  assert.equal(pickReactiveAction(frame, { rarest: 'oak_log', distance: 10, bearing: 30 }), 'turn_right')
  assert.equal(pickReactiveAction(frame, { rarest: 'oak_log', distance: 10, bearing: -30 }), 'turn_left')
  assert.equal(pickReactiveAction(frame, { rarest: 'oak_log', distance: 10, bearing: 0 }), 'forward')
})

test('loom and collision produce escape / jump', () => {
  const frame = { objectChannels: { LC4: 0.8 }, touchHead: 0, touchLegs: 0, odorBearingDeg: null }
  assert.equal(pickReactiveAction(frame, {}), 'turn_left')
  assert.equal(pickReactiveAction({ objectChannels: {}, touchHead: 0.8, touchLegs: 0, odorBearingDeg: null }, {}), 'jump')
})

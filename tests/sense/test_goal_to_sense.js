const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { loadSenseConfig, loadGoalSpec } = require('../../js/sense/loadConfig')
const { sampleWorld } = require('../../js/sense/senseBridge')
const { applyGoal, goalSuccess, rewardFor } = require('../../js/sense/goalToSense')
const { emptyFrame } = require('../../js/sense/frame')

const root = path.join(__dirname, '../..')
const cfg = loadSenseConfig(root)

const emptyWorld = {
  position: { x: 0, y: 64, z: 0 },
  onGround: true,
  blocks: [],
  items: [],
  entities: [],
}

test('empty world has no odor without a GoalSpec', () => {
  const worldFrame = sampleWorld({ ...emptyWorld, yawDeg: 0, velocity: { x: 0, y: 0, z: 0 }, dayFactor: 1 }, cfg)
  assert.equal(Object.keys(worldFrame.odor).length, 0)
})

test('oak and spruce GoalSpecs inject different glomerular tonics', () => {
  const worldFrame = sampleWorld({ ...emptyWorld, yawDeg: 0, velocity: { x: 0, y: 0, z: 0 }, dayFactor: 1 }, cfg)
  const oak = applyGoal(worldFrame, loadGoalSpec('collect_oak', root), emptyWorld)
  const spruce = applyGoal(worldFrame, loadGoalSpec('collect_spruce', root), emptyWorld)
  assert.ok((oak.odor.DL5 || 0) > (worldFrame.odor.DL5 || 0))
  assert.ok((oak.odor.DM1 || 0) > 0)
  assert.ok((spruce.odor.DC2 || 0) > 0)
})

test('feed GoalSpec injects sugar GRN tonic', () => {
  const hungry = applyGoal(emptyFrame(64), loadGoalSpec('feed', root), emptyWorld)
  assert.ok((hungry.taste.LB3b || 0) > 0)
})

test('flee:creeper only boosts LC4 / DA2 when a creeper is present', () => {
  const flee = loadGoalSpec('flee_creeper', root)
  const noCreeper = applyGoal(emptyFrame(64), flee, emptyWorld)
  assert.equal(noCreeper.objectChannels.LC4 || 0, 0)
  const near = applyGoal(emptyFrame(64), flee, {
    position: { x: 0, y: 64, z: 0 },
    entities: [{ id: 9, type: 'creeper', name: 'creeper', x: 3, y: 64, z: 0 }],
  })
  assert.ok(near.objectChannels.LC4 > 0)
  assert.ok((near.odor.DA2 || 0) > 0)
})

test('GoalSpec.success is for eval, not a network input', () => {
  const oakGoal = loadGoalSpec('collect_oak', root)
  assert.ok(goalSuccess(oakGoal, { inventory: { oak_log: 1 } }))
  assert.ok(!goalSuccess(oakGoal, { inventory: {} }))
  assert.ok(rewardFor(oakGoal, { inventory: {} }, { inventory: { oak_log: 1 } }) >= 10)
})

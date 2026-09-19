const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { loadSenseConfig } = require('../../js/sense/loadConfig')
const { sampleWorld } = require('../../js/sense/senseBridge')

const cfg = loadSenseConfig(path.join(__dirname, '../..'))

function baseWorld (over = {}) {
  return {
    tick: 1,
    dtS: 0.05,
    position: { x: 0, y: 64, z: 0 },
    yawDeg: 0,
    pitchDeg: 0,
    velocity: { x: 0, y: 0, z: 0 },
    onGround: true,
    inWater: false,
    raining: false,
    temperature: 0.8,
    dayFactor: 1,
    blocks: [],
    items: [],
    entities: [],
    ...over,
  }
}

test('cake block drives VA2 and DM1 (yeast / fruit esters)', () => {
  const cake = sampleWorld(baseWorld({
    blocks: [{ name: 'cake', x: 2, y: 64, z: 0, light: 12 }],
  }), cfg)
  assert.ok((cake.odor.VA2 || 0) > 0.02)
  assert.ok((cake.odor.DM1 || 0) > 0)
})

test('torch drives V / VP2 and lights retina columns', () => {
  const torch = sampleWorld(baseWorld({
    blocks: [{ name: 'torch', x: 1, y: 65, z: 0, light: 15 }],
  }), cfg)
  assert.ok((torch.odor.V || 0) > 0.05)
  assert.ok((torch.odor.VP2 || 0) > 0)
  assert.ok(torch.luminance.some(v => v > 0.2))
})

test('rain loads groom dust, moist, and water GRN LB3a', () => {
  const rain = sampleWorld(baseWorld({ raining: true, onGround: true }), cfg)
  assert.ok(rain.groomDust > 0.5)
  assert.ok(rain.moist > 0.5)
  assert.ok((rain.taste.LB3a || 0) > 0)
})

test('approaching player expands and drives loom / V', () => {
  const sprint = sampleWorld(baseWorld({
    tick: 2,
    entities: [{ id: 1, type: 'player', x: 1, y: 64, z: 2, w: 0.6, h: 1.8, d: 0.6 }],
  }), cfg, { prevObjects: { 1: { size: 8, az: 0, el: 0, tick: 1 } }, prevYaw: 0 })
  assert.equal(sprint.objects.length, 1)
  assert.ok(sprint.objects[0].expansionDegPerS > 0)
  assert.ok(sprint.objectChannels.LC4 > 0 || sprint.objectChannels.LC11 > 0)
  assert.ok((sprint.odor.V || 0) > 0)
})

test('spider eye tarsal bitter and cake labellar sugar', () => {
  const bitter = sampleWorld(baseWorld({ standingOn: 'spider_eye', onGround: true }), cfg)
  assert.ok((bitter.taste.LgAG1 || 0) > 0.5)
  const sugar = sampleWorld(baseWorld({ standingOn: 'cake', proboscisOut: true }), cfg)
  assert.ok((sugar.taste.LB3b || 0) > 0.5)
})

test('self-motion JO wind and collision bristles', () => {
  const wind = sampleWorld(baseWorld({ velocity: { x: 0, y: 0, z: 0.4 } }), cfg)
  assert.ok(wind.windLeft > 0 && wind.windRight > 0)
  const collide = sampleWorld(baseWorld({ horizontalCollision: true }), cfg)
  assert.ok(collide.touchHead > 0.5)
})

const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { loadSenseConfig } = require('../../js/sense/loadConfig')
const { sampleWorld, retinaDirections, retinaLayout } = require('../../js/sense/senseBridge')

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

// The eye is no longer a grid, so there are no rows, columns or diagonals to check.
// What is worth checking is that the directions really are the measured ones and that
// each eye looks at its own side of the world.

test('rays are measured ommatidia, one per sampled column, not a synthetic grid', () => {
  const dirs = retinaDirections(cfg.retina)
  assert.equal(dirs.length, cfg.retina.columns)
  assert.equal(dirs.length, cfg.retina.rays.length)
  for (const dir of dirs) {
    // Every ray names the ommatidium it came from, so it can be traced to a column.
    assert.ok(Number.isInteger(dir.hex1) && Number.isInteger(dir.hex2))
    assert.ok(dir.eye === 'L' || dir.eye === 'R')
    const norm = Math.hypot(dir.dx, dir.dy, dir.dz)
    assert.ok(Math.abs(norm - 1) < 1e-6, 'direction must be a unit vector')
  }
  // Two ommatidia must not share a direction, or the eye has duplicate pixels.
  const keys = new Set(dirs.map(d => `${d.eye}:${d.hex1}:${d.hex2}`))
  assert.equal(keys.size, dirs.length)
})

test('each eye looks at its own hemifield, meeting near the midline', () => {
  const dirs = retinaDirections(cfg.retina)
  // Azimuth degenerates near the poles, so judge the field in the horizontal band.
  const band = dirs.filter(d => Math.abs(d.el) < 20)
  const left = band.filter(d => d.eye === 'L').map(d => d.az)
  const right = band.filter(d => d.eye === 'R').map(d => d.az)
  assert.ok(left.length > 0 && right.length > 0)
  // + is the fly's right, matching odorBearingDeg.
  assert.ok(Math.min(...left) < -100, 'left eye must see well behind on its own side')
  assert.ok(Math.max(...left) < 20, 'left eye must not sweep across to the right')
  assert.ok(Math.max(...right) > 100, 'right eye must see well behind on its own side')
  assert.ok(Math.min(...right) > -20, 'right eye must not sweep across to the left')
  const overlap = Math.max(...left) - Math.min(...right)
  assert.ok(overlap > 0 && overlap < 40, `frontal overlap should be narrow, got ${overlap}`)
})

test('rays are split evenly between the eyes', () => {
  const { perEye, eyes, count } = retinaLayout(cfg.retina)
  assert.deepEqual(eyes.slice().sort(), ['L', 'R'])
  const dirs = retinaDirections(cfg.retina)
  for (const eye of eyes) {
    assert.equal(dirs.filter(d => d.eye === eye).length, perEye)
  }
  assert.equal(count, perEye * eyes.length)
})

test('a missing generated direction file is an error, not a silent fallback', () => {
  assert.throws(() => retinaDirections({ maxDistance: 24 }), /retina_columns\.json/)
})

test('a tree straight ahead darkens frontal columns in both eyes', () => {
  const trunk = []
  for (let y = 63; y <= 68; y++) {
    // yaw 0 faces +z in Minecraft, so a tree "ahead" is at positive z
    for (const x of [-1, 0]) trunk.push({ name: 'oak_log', x, y, z: 4, light: 12 })
  }
  const seen = sampleWorld(baseWorld({ blocks: trunk }), cfg)
  const dirs = retinaDirections(cfg.retina)
  const { perEye } = retinaLayout(cfg.retina)
  const frontal = eye => dirs
    .map((d, i) => ({ d, i }))
    .filter(({ d }) => d.eye === eye && Math.abs(d.az) <= 25 && Math.abs(d.el) <= 25)
  for (const eye of ['L', 'R']) {
    const idx = frontal(eye)
    assert.ok(idx.length > 0, `${eye} eye must have frontal rays`)
    const lit = idx.some(({ i }) => seen.luminance[i] > 0 && seen.luminance[i] < 0.5)
    assert.ok(lit, `${eye} eye should register the trunk ahead`)
  }
  assert.equal(dirs.filter(d => d.eye === 'L').length, perEye)
})

test('self-motion JO wind and collision bristles', () => {
  const wind = sampleWorld(baseWorld({ velocity: { x: 0, y: 0, z: 0.4 } }), cfg)
  assert.ok(wind.windLeft > 0 && wind.windRight > 0)
  const collide = sampleWorld(baseWorld({ horizontalCollision: true }), cfg)
  assert.ok(collide.touchHead > 0.5)
})

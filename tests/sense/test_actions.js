const { test } = require('node:test')
const assert = require('node:assert/strict')
const { ACTIONS, actionIndex, applyAction, isCreative, motorFor, releaseControls } = require('../../js/sense/actions')

function fakeBot () {
  const states = {}
  return {
    states,
    looks: [],
    digs: 0,
    stops: 0,
    entity: { position: { x: 0, y: 64, z: 0 }, yaw: 0, pitch: 0 },
    game: { gameMode: 'creative' },
    setControlState (key, on) { states[key] = on },
    async look (yaw, pitch) { this.looks.push({ yaw, pitch }); this.entity.yaw = yaw; this.entity.pitch = pitch },
    blockAtCursor () { return { name: 'oak_log' } },
    async dig () { this.digs++ ; await new Promise(r => setTimeout(r, 5)) },
    stopDigging () { this.stops++ },
  }
}

test('action list and index are stable', () => {
  assert.equal(ACTIONS.length, 9)
  assert.equal(actionIndex('mine'), 5)
  assert.equal(actionIndex('nonsense'), ACTIONS.indexOf('noop'))
})

test('forward is a held command, not a timed pulse', async () => {
  const bot = fakeBot()
  await applyAction(bot, 'forward', {})
  assert.equal(bot.states.forward, true)
  // A later turn steers without stopping the walk, like a course correction.
  await applyAction(bot, 'turn_left', { turnDeg: 12 })
  assert.equal(bot.states.forward, true, 'turning must not cancel locomotion')
  assert.ok(bot.entity.yaw > 0)
})

test('noop is the absence of a command and stops the legs', async () => {
  const bot = fakeBot()
  await applyAction(bot, 'forward', {})
  await applyAction(bot, 'noop', {})
  assert.equal(bot.states.forward, false)
  assert.equal(bot.states.back, false)
})

test('a 15 Hz step never blocks', async () => {
  const bot = fakeBot()
  const t0 = Date.now()
  for (const name of ACTIONS) await applyAction(bot, name, {})
  assert.ok(Date.now() - t0 < 67, `all nine actions should fit inside one 15 Hz tick, took ${Date.now() - t0}ms`)
})

test('mine starts one dig and holds it across ticks, then releases', async () => {
  const bot = fakeBot()
  await applyAction(bot, 'mine', {})
  await applyAction(bot, 'mine', {})
  assert.equal(bot.digs, 1, 'repeating mine must not restart the dig')
  assert.equal(motorFor(bot).digging, true)
  await applyAction(bot, 'forward', {})
  assert.equal(bot.stops, 1, 'a different command stops digging')
})

test('jump is momentary', async () => {
  const bot = fakeBot()
  await applyAction(bot, 'jump', {})
  assert.equal(bot.states.jump, true)
  await applyAction(bot, 'forward', {})
  assert.equal(bot.states.jump, false)
})

test('camera actions stay inside the pitch limits', async () => {
  const bot = fakeBot()
  for (let i = 0; i < 40; i++) await applyAction(bot, 'camera_up', { pitchDeg: 8 })
  assert.ok(bot.entity.pitch >= -Math.PI / 2 - 1e-9)
  for (let i = 0; i < 80; i++) await applyAction(bot, 'camera_down', { pitchDeg: 8 })
  assert.ok(bot.entity.pitch <= Math.PI / 2 + 1e-9)
})

test('disconnected bot does not throw', async () => {
  assert.equal(isCreative(null), false)
  assert.equal(isCreative({}), false)
  await applyAction(null, 'forward')
  releaseControls(null)
})

test('releaseControls clears every state', async () => {
  const bot = fakeBot()
  await applyAction(bot, 'forward', {})
  releaseControls(bot)
  assert.equal(bot.states.forward, false)
  assert.equal(bot.states.sprint, false)
  assert.equal(motorFor(bot).locomotion, 'stop')
})

function wingedBot () {
  const bot = fakeBot()
  bot.creative = {
    flying: false,
    startFlying () { this.flying = true },
    stopFlying () { this.flying = false },
  }
  return bot
}

test('forward on the ground is a walk, not an automatic takeoff', async () => {
  const bot = wingedBot()
  const pos = { ...bot.entity.position }
  await applyAction(bot, 'forward', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, false)
  assert.equal(bot.states.forward, true)
  assert.equal(bot.entity.position.x, pos.x)
  assert.equal(bot.entity.position.y, pos.y)
  assert.equal(bot.entity.position.z, pos.z)
})

test('after takeoff, forward flies along the gaze', async () => {
  const bot = wingedBot()
  bot.entity.yaw = 0
  bot.entity.pitch = 0
  await applyAction(bot, 'jump', { flySpeed: 6, dtMs: 1000 })
  const z0 = bot.entity.position.z
  const y0 = bot.entity.position.y
  await applyAction(bot, 'forward', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, true)
  assert.ok(bot.entity.position.z !== z0, 'airborne DNp09 has to cover air')
  assert.equal(bot.entity.position.y, y0, 'level gaze stays level')
})

test('jump is takeoff: the body climbs and does not start a walk', async () => {
  const bot = wingedBot()
  const y0 = bot.entity.position.y
  await applyAction(bot, 'jump', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.states.jump, true)
  assert.equal(bot.states.forward, false)
  assert.equal(motorFor(bot).locomotion, 'stop')
  assert.equal(bot.creative.flying, true)
  assert.ok(bot.entity.position.y > y0, 'DNp01 is takeoff, so the body goes up')
})

test('halt lands: gravity comes back and the wings stop', async () => {
  const bot = wingedBot()
  await applyAction(bot, 'jump', { flySpeed: 6, dtMs: 1000 })
  await applyAction(bot, 'forward', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, true)
  const frozen = { ...bot.entity.position }
  await applyAction(bot, 'noop', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, false)
  assert.equal(motorFor(bot).airborne, false)
  assert.equal(bot.states.forward, false)
  assert.equal(bot.entity.position.x, frozen.x)
  assert.equal(bot.entity.position.y, frozen.y)
  assert.equal(bot.entity.position.z, frozen.z)
})

test('a landed fly can take off again', async () => {
  const bot = wingedBot()
  await applyAction(bot, 'jump', { flySpeed: 6, dtMs: 1000 })
  await applyAction(bot, 'noop', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, false)
  const y0 = bot.entity.position.y
  await applyAction(bot, 'jump', { flySpeed: 6, dtMs: 1000 })
  assert.equal(bot.creative.flying, true)
  assert.ok(bot.entity.position.y > y0)
})

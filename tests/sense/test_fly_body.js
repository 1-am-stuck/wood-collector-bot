const { test } = require('node:test')
const assert = require('node:assert/strict')
const {
  FLYBODY_LENGTH_M,
  VANILLA_SCALE_MIN,
  targetScale,
  appliedScale,
  standingHeightM,
  standingWidthM,
  eyeHeightM,
  reachBlocks,
  attributeCommands,
  applyFlyBody,
} = require('../../js/sense/flyBody')

test('flybody length is the latest MuJoCo Drosophila (2.97 mm)', () => {
  // TuragaLab/flybody, Nature 2025: full body length 0.297 cm, wingspan 0.604 cm.
  assert.equal(FLYBODY_LENGTH_M, 0.00297)
})

test('target scale maps that length onto the 1.8 m player hitbox', () => {
  assert.ok(Math.abs(targetScale() - 0.00297 / 1.8) < 1e-12)
  assert.ok(targetScale() < 0.002)
})

test('vanilla minecraft:scale cannot go below 0.0625, so we clamp', () => {
  assert.equal(VANILLA_SCALE_MIN, 0.0625)
  assert.equal(appliedScale(), VANILLA_SCALE_MIN)
  assert.ok(appliedScale() > targetScale())
  assert.ok(standingHeightM() < 0.12, 'must be insect-scale vs a 1 m block')
  assert.ok(standingWidthM() < 0.04)
  assert.ok(eyeHeightM() < 0.11)
})

test('dig / labellum reach is contact, not a 5-block player arm', () => {
  const reach = reachBlocks(5)
  assert.ok(reach < 0.5)
  assert.ok(reach >= 0.25)
})

test('attribute commands set scale and shrink block reach', () => {
  const cmds = attributeCommands()
  assert.ok(cmds.some(c => c.includes('minecraft:scale') && c.includes('0.0625')))
  assert.ok(cmds.some(c => c.includes('block_interaction_range')))
})

test('applyFlyBody chats those commands at the server', async () => {
  const chats = []
  const bot = { chat (s) { chats.push(s) } }
  await applyFlyBody(bot)
  assert.ok(chats.length >= 2)
  assert.ok(chats.every(c => c.startsWith('/attribute')))
})

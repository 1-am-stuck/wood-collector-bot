const { test } = require('node:test')
const assert = require('node:assert/strict')
const { Vec3 } = require('vec3')
const { sampleWorldView, sampleEyeView, colorForBlock, orbitEye, packRgb, packLum } = require('../../js/sense/eyeView')

// A 3-wide oak trunk 4 blocks north (+z is forward at yaw 0) on a grass floor.
function fakeBot (pos = new Vec3(0.5, 64, 0.5)) {
  return {
    entity: { position: pos, yaw: 0, pitch: 0, eyeHeight: 1.62, height: 1.8 },
    blockAt (p) {
      if (p.y === 63) return { name: 'grass_block', boundingBox: 'block', position: p }
      if (p.y > 63 && p.y < 69 && p.z >= 3 && p.z <= 5 && p.x >= -1 && p.x <= 1) {
        return { name: 'oak_log', boundingBox: 'block', position: p }
      }
      return { name: 'air', boundingBox: 'empty', position: p }
    },
  }
}

test('oak log is a warm pixel, sky is cool', () => {
  const oak = colorForBlock('oak_log', 4, 28, { axis: 'z', step: 1, vx: 0, vy: 64, vz: 4 })
  assert.ok(oak[0] > oak[2], 'log should be warmer than blue')
  const sky = colorForBlock(null, 24)
  assert.ok(sky[2] >= sky[0], 'sky should be cool')
})

test('faces are shaded so flat terrain is not one solid block of colour', () => {
  const top = colorForBlock('grass_block', 6, 32, { axis: 'y', step: -1, vx: 0, vy: 63, vz: 0 })
  const side = colorForBlock('grass_block', 6, 32, { axis: 'z', step: 1, vx: 0, vy: 63, vz: 0 })
  const under = colorForBlock('grass_block', 6, 32, { axis: 'y', step: 1, vx: 0, vy: 63, vz: 0 })
  assert.ok(top[1] > side[1], 'a top face is brighter than a side face')
  assert.ok(side[1] > under[1], 'a side face is brighter than an underside')
})

test('first-person view paints the trunk the fly is facing', () => {
  const view = sampleEyeView(fakeBot(), { width: 32, height: 18, maxDist: 24 })
  assert.equal(view.mode, 'first')
  assert.equal(view.rgb.length, 32 * 18 * 3)
  const centre = ((9 * 32) + 16) * 3
  assert.ok(view.rgb[centre] > view.rgb[centre + 2], 'centre pixel should be the warm trunk')
})

test('orbit camera sits behind the bot and looks back at it', () => {
  const bot = fakeBot()
  const cam = orbitEye(bot, { camYawDeg: 0, camPitchDeg: 0, distance: 8 })
  assert.ok(Math.abs(cam.distance - 8) < 1e-6)
  // Camera at +z of the bot, so its forward vector points back along -z.
  assert.ok(cam.forward.z < -0.9, 'camera must look at the bot')
  const view = sampleWorldView(bot, { mode: 'orbit', width: 16, height: 9, camYawDeg: 0, distance: 8 })
  assert.equal(view.mode, 'orbit')
  assert.deepEqual(view.botScreen, { x: 0.5, y: 0.5 }, 'the bot is centred by construction')
})

test('orbit camera pulls in rather than sitting inside terrain', () => {
  const bot = fakeBot()
  // Solid everywhere except the bot's own column: a far camera has nowhere to go.
  bot.blockAt = p => (Math.abs(p.x - 0.5) < 1.5 && Math.abs(p.z - 0.5) < 1.5 && p.y > 63
    ? { name: 'air', boundingBox: 'empty', position: p }
    : { name: 'stone', boundingBox: 'block', position: p })
  const cam = orbitEye(bot, { camYawDeg: 0, camPitchDeg: 0, distance: 20 })
  assert.ok(cam.distance < 20, 'camera should have been pulled in')
  assert.ok(cam.distance >= 1.5, 'camera never collapses onto the bot')
})

test('camera direction actually changes what is rendered', () => {
  const bot = fakeBot()
  const front = sampleWorldView(bot, { mode: 'orbit', width: 16, height: 9, camYawDeg: 0, distance: 6 })
  const behind = sampleWorldView(bot, { mode: 'orbit', width: 16, height: 9, camYawDeg: 180, distance: 6 })
  assert.notDeepEqual(front.rgb, behind.rgb, 'a different camera yaw must give a different image')
})

test('pack helpers stay byte-sized', () => {
  assert.equal(Buffer.from(packRgb([1, 2, 3, 4]), 'base64').length, 4)
  assert.equal(Buffer.from(packLum([0, 1, 0.5]), 'base64')[1], 255)
})

test('packEye is the LLM-sized first-person frame', () => {
  const { packEye } = require('../../js/sense/eyeView')
  const view = sampleEyeView(fakeBot(), { width: 8, height: 4, maxDist: 24 })
  const packed = packEye(view)
  assert.equal(packed.w, 8)
  assert.equal(packed.h, 4)
  assert.equal(packed.mode, 'first')
  assert.equal(Buffer.from(packed.rgb, 'base64').length, 8 * 4 * 3)
})

const { test } = require('node:test')
const assert = require('node:assert/strict')
const { Vec3 } = require('vec3')
const { sampleEyeView, sampleFlyRetina, colorForBlock, packRgb, packLum } = require('../../js/sense/eyeView')

test('oak log is a brown visual-cortex pixel, sky is blue-ish', () => {
  const oak = colorForBlock('oak_log', 4)
  assert.ok(oak[0] > oak[2], 'log should be warmer than blue')
  const sky = colorForBlock(null, 24)
  assert.ok(sky[2] >= sky[0], 'sky should be cool')
})

test('sampleEyeView paints the block the fly is looking at', () => {
  const origin = new Vec3(0, 64, 0)
  const bot = {
    entity: {
      position: origin,
      yaw: 0,
      pitch: 0,
      eyeHeight: 1.62,
    },
    blockAt (pos) {
      if (pos.z <= -2 && pos.z >= -5 && Math.abs(pos.x) < 2 && pos.y >= 63 && pos.y <= 67) {
        return { name: 'oak_log', boundingBox: 'block', position: new Vec3(0, 64, -3) }
      }
      return { name: 'air', boundingBox: 'empty', position: pos }
    },
  }
  const eye = sampleEyeView(bot, { width: 16, height: 9, maxDist: 16 })
  assert.equal(eye.w, 16)
  assert.equal(eye.h, 9)
  assert.equal(eye.rgb.length, 16 * 9 * 3)
  const mid = ((4 * 16) + 8) * 3
  assert.ok(eye.rgb[mid] > 40, 'center pixel should hit the oak in front')
})

test('sampleFlyRetina is a 64x64 luminance grid and pack helpers stay byte-sized', () => {
  const origin = new Vec3(0, 64, 0)
  const bot = {
    entity: { position: origin, yaw: 0, pitch: 0, eyeHeight: 1.62 },
    blockAt (pos) {
      if (pos.z <= -2 && pos.z >= -5 && Math.abs(pos.x) < 2 && pos.y >= 63 && pos.y <= 67) {
        return { name: 'oak_log', boundingBox: 'block', position: new Vec3(0, 64, -3) }
      }
      return { name: 'air', boundingBox: 'empty', position: pos }
    },
  }
  const fly = sampleFlyRetina(bot, { width: 8, height: 8, maxDist: 16 })
  assert.equal(fly.w, 8)
  assert.equal(fly.lum.length, 64)
  assert.ok(fly.lum.some(v => v > 0.2))
  const rgb = packRgb([1, 2, 3, 4])
  assert.equal(Buffer.from(rgb, 'base64').length, 4)
  assert.equal(Buffer.from(packLum([0, 1, 0.5]), 'base64')[1], 255)
})

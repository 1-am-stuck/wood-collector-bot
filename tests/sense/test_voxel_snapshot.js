const { test } = require('node:test')
const assert = require('node:assert/strict')
const { Vec3 } = require('vec3')
const { sampleVoxels, samplePose, snapshotStale } = require('../../js/sense/voxelSnapshot')

// Solid slab at y=63, a 2x2 trunk above it, air everywhere else.
function fakeBot (pos = new Vec3(0.5, 64, 0.5)) {
  const registry = {
    blocksByStateId: [
      { name: 'air' },
      { name: 'grass_block' },
      { name: 'oak_log' },
    ],
  }
  return {
    entity: {
      position: pos,
      yaw: 0,
      pitch: 0,
      height: 1.8,
      onGround: true,
      velocity: new Vec3(0, 0, 0),
    },
    entities: {},
    registry,
    world: {
      getBlockStateId (p) {
        if (p.y === 63) return 1
        if (p.y > 63 && p.y < 68 && (p.x === 0 || p.x === 1) && (p.z === 4 || p.z === 5)) return 2
        return 0
      },
    },
  }
}

function unpack (snapshot) {
  const offsets = Buffer.from(snapshot.offsets, 'base64')
  const states = Buffer.from(snapshot.states, 'base64')
  const out = []
  for (let i = 0; i < snapshot.count; i++) {
    out.push({
      x: snapshot.origin.x + offsets[i * 3],
      y: snapshot.origin.y + offsets[i * 3 + 1],
      z: snapshot.origin.z + offsets[i * 3 + 2],
      name: snapshot.palette[states.readUInt16LE(i * 2)],
    })
  }
  return out
}

test('snapshot keeps surface blocks with their real names', () => {
  const snap = sampleVoxels(fakeBot(), { radiusXZ: 8, radiusY: 6 })
  assert.equal(snap.dims.x, 17)
  assert.equal(snap.dims.y, 13)
  assert.ok(snap.count > 0)
  const voxels = unpack(snap)
  assert.ok(voxels.some(v => v.name === 'oak_log'), 'trunk should be in the snapshot')
  assert.ok(voxels.some(v => v.name === 'grass_block'), 'ground should be in the snapshot')
  assert.ok(!voxels.some(v => v.name === 'air'), 'air is never sent')
  const trunk = voxels.filter(v => v.name === 'oak_log')
  assert.ok(trunk.every(v => v.z === 4 || v.z === 5), 'trunk stays where the world put it')
})

test('interior blocks are culled, surfaces are not', () => {
  // A solid 5x5x5 cube: only its shell can ever be seen.
  const bot = fakeBot()
  bot.world.getBlockStateId = p => (
    p.x >= -2 && p.x <= 2 && p.y >= 62 && p.y <= 66 && p.z >= -2 && p.z <= 2 ? 1 : 0
  )
  const snap = sampleVoxels(bot, { radiusXZ: 6, radiusY: 6 })
  const solid = 5 * 5 * 5
  const shell = solid - 3 * 3 * 3
  assert.equal(snap.count, shell, 'only the shell survives culling')
  assert.ok(snap.count < solid)
})

test('pose is small and carries what a camera needs', () => {
  const pose = samplePose(fakeBot())
  assert.equal(pose.yaw, 0)
  assert.equal(pose.onGround, true)
  assert.ok(Math.abs(pose.x - 0.5) < 1e-9)
  assert.deepEqual(pose.mobs, [])
  assert.ok(JSON.stringify(pose).length < 400, 'pose must stay cheap at 30 Hz')
})

test('snapshot goes stale once the bot walks out of the box', () => {
  const bot = fakeBot()
  const snap = sampleVoxels(bot, { radiusXZ: 8, radiusY: 6 })
  assert.equal(snapshotStale(snap, bot), false)
  bot.entity.position = new Vec3(20.5, 64, 0.5)
  assert.equal(snapshotStale(snap, bot), true)
  assert.equal(snapshotStale(null, bot), true)
})

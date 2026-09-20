/** Compact block grid around the bot, for rendering in the browser.
 *
 * Ray-casting a camera image on the server costs one walk per pixel, which does
 * not fit in a 30 Hz budget. Instead we ship the neighbourhood once (and again
 * when the bot has moved), and let WebGL draw it at whatever rate and from
 * whatever angle the viewer wants. Camera control then costs no round trip.
 *
 * Only *surface* voxels are sent: a voxel with at least one transparent
 * neighbour. Interior blocks can never be seen, so they are dropped, which is
 * usually an order of magnitude less data.
 *
 * Like eyeView, this is for us. The policy never receives it.
 */

const { Vec3 } = require('vec3')

const AIR_NAMES = new Set(['air', 'cave_air', 'void_air'])

/** State ids that count as see-through, resolved once per bot from the registry. */
function transparentStateIds (bot) {
  const ids = new Set([0])
  const registry = bot && bot.registry
  const byState = registry && registry.blocksByStateId
  if (!byState) return ids
  for (let id = 0; id < byState.length; id++) {
    const block = byState[id]
    if (block && AIR_NAMES.has(block.name)) ids.add(id)
  }
  return ids
}

/** Cheap state lookup: avoids allocating a Block per voxel where possible. */
function stateReader (bot) {
  const world = bot && bot.world
  const cursor = new Vec3(0, 0, 0)
  if (world && typeof world.getBlockStateId === 'function') {
    return (x, y, z) => {
      cursor.x = x; cursor.y = y; cursor.z = z
      const id = world.getBlockStateId(cursor)
      return id == null ? 0 : id
    }
  }
  return (x, y, z) => {
    cursor.x = x; cursor.y = y; cursor.z = z
    const block = bot.blockAt(cursor, false)
    if (!block) return 0
    return block.stateId == null ? (AIR_NAMES.has(block.name) ? 0 : 1) : block.stateId
  }
}

function nameForState (bot, id) {
  const byState = bot && bot.registry && bot.registry.blocksByStateId
  const block = byState && byState[id]
  if (block && block.name) return block.name
  return id === 0 ? 'air' : `state_${id}`
}

/**
 * @param {object} bot Mineflayer bot
 * @param {object} [opts] `{ radiusXZ, radiusY }`
 * @returns {object|null} `{ origin, dims, count, offsets, states, palette }`
 */
function sampleVoxels (bot, opts = {}) {
  if (!bot || !bot.entity) return null
  const rxz = Math.max(4, Math.min(80, opts.radiusXZ == null ? 24 : opts.radiusXZ))
  const ry = Math.max(4, Math.min(64, opts.radiusY == null ? 14 : opts.radiusY))
  const pos = bot.entity.position
  const ox = Math.floor(pos.x) - rxz
  const oy = Math.floor(pos.y) - ry
  const oz = Math.floor(pos.z) - rxz
  const dx = rxz * 2 + 1
  const dy = ry * 2 + 1
  const dz = rxz * 2 + 1

  const readState = stateReader(bot)
  const clear = transparentStateIds(bot)
  const states = new Uint16Array(dx * dy * dz)
  const at = (x, y, z) => states[(y * dz + z) * dx + x]
  for (let y = 0; y < dy; y++) {
    for (let z = 0; z < dz; z++) {
      const row = (y * dz + z) * dx
      for (let x = 0; x < dx; x++) {
        states[row + x] = readState(ox + x, oy + y, oz + z)
      }
    }
  }

  const offsets = []
  const ids = []
  const palette = {}
  for (let y = 0; y < dy; y++) {
    for (let z = 0; z < dz; z++) {
      for (let x = 0; x < dx; x++) {
        const id = at(x, y, z)
        if (clear.has(id)) continue
        // Edges of the box count as exposed so the shell is never hollow.
        const exposed =
          x === 0 || x === dx - 1 || y === 0 || y === dy - 1 || z === 0 || z === dz - 1 ||
          clear.has(at(x - 1, y, z)) || clear.has(at(x + 1, y, z)) ||
          clear.has(at(x, y - 1, z)) || clear.has(at(x, y + 1, z)) ||
          clear.has(at(x, y, z - 1)) || clear.has(at(x, y, z + 1))
        if (!exposed) continue
        offsets.push(x, y, z)
        ids.push(id)
        if (palette[id] === undefined) palette[id] = nameForState(bot, id)
      }
    }
  }

  return {
    origin: { x: ox, y: oy, z: oz },
    dims: { x: dx, y: dy, z: dz },
    count: ids.length,
    offsets: Buffer.from(Uint8Array.from(offsets)).toString('base64'),
    states: Buffer.from(new Uint16Array(ids).buffer).toString('base64'),
    palette,
  }
}

/** Pose is tiny, so this is what actually goes out at 30 Hz. */
function samplePose (bot) {
  if (!bot || !bot.entity) return null
  const e = bot.entity
  const mobs = []
  for (const other of Object.values(bot.entities || {})) {
    if (!other || other === e || !other.position) continue
    const d = other.position.distanceTo(e.position)
    if (d > 32) continue
    mobs.push({
      id: other.id,
      kind: other.name || other.username || other.type || 'entity',
      x: other.position.x,
      y: other.position.y,
      z: other.position.z,
      h: other.height || 1,
      w: other.width || 0.6,
    })
    if (mobs.length >= 24) break
  }
  return {
    x: e.position.x,
    y: e.position.y,
    z: e.position.z,
    yaw: e.yaw,
    pitch: e.pitch,
    height: e.height || 1.8,
    width: e.width || 0.6,
    onGround: !!e.onGround,
    vx: (e.velocity && e.velocity.x) || 0,
    vy: (e.velocity && e.velocity.y) || 0,
    vz: (e.velocity && e.velocity.z) || 0,
    mobs,
  }
}

/** Has the bot moved far enough that the cached neighbourhood is stale? */
function snapshotStale (snapshot, bot, margin = 6) {
  if (!snapshot || !bot || !bot.entity) return true
  const { origin, dims } = snapshot
  const p = bot.entity.position
  const cx = origin.x + (dims.x - 1) / 2
  const cy = origin.y + (dims.y - 1) / 2
  const cz = origin.z + (dims.z - 1) / 2
  return (
    Math.abs(p.x - cx) > margin ||
    Math.abs(p.y - cy) > margin ||
    Math.abs(p.z - cz) > margin
  )
}

module.exports = { sampleVoxels, samplePose, snapshotStale, transparentStateIds, AIR_NAMES }

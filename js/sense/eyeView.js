/** Human-facing renders of the world FruitFly is standing in.
 *
 * Nothing here is an input to the policy. These are pictures for us, cast with
 * the same `blockAt` voxel rays the visual cortex uses, so what we look at is
 * the world the 64 luminance columns were sampled from.
 *
 * Two cameras:
 *  - `orbit`: third person, centred on the bot, direction and distance settable
 *  - `first`: down the bot's own look vector
 */

const { Vec3 } = require('vec3')

const BASE_COLORS = [
  [/oak_log|oak_wood/, [124, 88, 50]],
  [/spruce/, [78, 55, 36]],
  [/birch/, [196, 176, 128]],
  [/cherry/, [168, 92, 98]],
  [/acacia/, [150, 78, 42]],
  [/dark_oak/, [52, 36, 22]],
  [/jungle|mangrove/, [92, 62, 32]],
  [/pale_oak/, [186, 176, 164]],
  [/leaves|vine|moss/, [46, 110, 42]],
  [/grass_block|grass|fern/, [72, 130, 52]],
  [/dirt|podzol|mud|rooted/, [110, 78, 48]],
  [/sand|sandstone/, [210, 196, 130]],
  [/gravel/, [136, 130, 126]],
  [/water|kelp|seagrass/, [40, 90, 180]],
  [/lava|magma/, [214, 110, 40]],
  [/snow|powder_snow/, [228, 234, 240]],
  [/ice/, [160, 200, 232]],
  [/deepslate|basalt|blackstone/, [70, 70, 76]],
  [/stone|cobble|andesite|diorite|granite|ore|tuff/, [116, 116, 124]],
  [/plank|fence|door|stair|slab/, [162, 130, 84]],
  [/log|wood|stem/, [100, 72, 40]],
  [/flower|poppy|dandelion|tulip/, [190, 90, 110]],
]

const SKY = [126, 176, 232]

/** Which face we came in through, so surfaces separate instead of reading flat. */
function faceShade (axis, step) {
  if (axis === 'y') return step > 0 ? 0.52 : 1.0
  if (axis === 'x') return 0.82
  return 0.66
}

function baseColor (name) {
  const n = String(name || '').replace(/^minecraft:/, '')
  for (const [re, rgb] of BASE_COLORS) {
    if (re.test(n)) return rgb
  }
  return [96, 100, 78]
}

/**
 * @param {string|null} name block hit, or null for sky
 * @param {number} dist blocks travelled
 * @param {number} maxDist fog horizon
 * @param {object} [face] `{ axis, step, vx, vy, vz }` from the voxel walk
 */
function colorForBlock (name, dist, maxDist = 28, face = null) {
  const horizon = Math.max(1, maxDist)
  const fog = Math.min(1, (dist || 0) / horizon)
  if (!name) {
    const t = Math.max(0, Math.min(1, 0.35 + fog * 0.4))
    return [Math.round(70 + 90 * t), Math.round(120 + 70 * t), Math.round(190 + 40 * t)]
  }
  const [r0, g0, b0] = baseColor(name)
  let shade = face ? faceShade(face.axis, face.step) : 0.85
  if (face && face.vx != null) {
    // Tiny per-voxel offset so large flat areas still show block structure.
    const d = ((face.vx * 7 + face.vy * 13 + face.vz * 29) % 5 + 5) % 5
    shade *= 1 + (d - 2) * 0.014
  }
  const mix = 0.55 * fog
  const blend = (c, sky) => Math.max(0, Math.min(255, Math.round(c * shade * (1 - mix) + sky * mix)))
  return [blend(r0, SKY[0]), blend(g0, SKY[1]), blend(b0, SKY[2])]
}

function isSolid (block) {
  return !!block && block.name !== 'air' && block.boundingBox !== 'empty'
}

/** Amanatides-Woo voxel walk. Reports the entry face so we can shade it. */
function voxelHit (bot, from, dir, maxDist) {
  let x = Math.floor(from.x)
  let y = Math.floor(from.y)
  let z = Math.floor(from.z)
  const sx = dir.x > 0 ? 1 : dir.x < 0 ? -1 : 0
  const sy = dir.y > 0 ? 1 : dir.y < 0 ? -1 : 0
  const sz = dir.z > 0 ? 1 : dir.z < 0 ? -1 : 0
  const tdx = sx !== 0 ? Math.abs(1 / dir.x) : Infinity
  const tdy = sy !== 0 ? Math.abs(1 / dir.y) : Infinity
  const tdz = sz !== 0 ? Math.abs(1 / dir.z) : Infinity
  let tmaxX = sx !== 0 ? (sx > 0 ? (x + 1 - from.x) : (from.x - x)) * tdx : Infinity
  let tmaxY = sy !== 0 ? (sy > 0 ? (y + 1 - from.y) : (from.y - y)) * tdy : Infinity
  let tmaxZ = sz !== 0 ? (sz > 0 ? (z + 1 - from.z) : (from.z - z)) * tdz : Infinity
  let t = 0
  let axis = 'y'
  let step = sy
  while (t <= maxDist) {
    const b = bot.blockAt(new Vec3(x, y, z), false)
    if (isSolid(b)) {
      return { name: b.name, dist: t, axis, step, vx: x, vy: y, vz: z }
    }
    if (tmaxX < tmaxY) {
      if (tmaxX < tmaxZ) {
        t = tmaxX; tmaxX += tdx; x += sx; axis = 'x'; step = sx
      } else {
        t = tmaxZ; tmaxZ += tdz; z += sz; axis = 'z'; step = sz
      }
    } else if (tmaxY < tmaxZ) {
      t = tmaxY; tmaxY += tdy; y += sy; axis = 'y'; step = sy
    } else {
      t = tmaxZ; tmaxZ += tdz; z += sz; axis = 'z'; step = sz
    }
  }
  return null
}

function hitAlongRay (bot, from, dir, maxDist) {
  if (typeof bot.blockAt === 'function') return voxelHit(bot, from, dir, maxDist)
  if (bot.world && typeof bot.world.raycast === 'function') {
    const hit = bot.world.raycast(from, dir, maxDist, blk => blk && blk.boundingBox !== 'empty')
    if (!hit) return null
    const pos = hit.position || from
    return {
      name: hit.name,
      dist: typeof from.distanceTo === 'function' ? from.distanceTo(pos) : maxDist,
      axis: 'y',
      step: -1,
    }
  }
  return null
}

function norm (v) {
  const len = Math.hypot(v.x, v.y, v.z) || 1
  return new Vec3(v.x / len, v.y / len, v.z / len)
}

function cross (a, b) {
  return new Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

/**
 * Minecraft look vector: yaw 0 faces +z, positive pitch looks down. This must
 * match `forward()` in senseBridge, or the picture we show is not the direction
 * the fly's retina was sampled from.
 */
function lookDir (yaw, pitch) {
  const cp = Math.cos(pitch)
  return new Vec3(-Math.sin(yaw) * cp, -Math.sin(pitch), Math.cos(yaw) * cp)
}

/**
 * Place a third-person camera on a sphere around the bot. If the requested spot
 * is inside terrain we pull the camera in until it is in open air, the way any
 * orbit camera has to.
 */
function orbitEye (bot, opts) {
  const body = bot.entity.position
  const target = new Vec3(body.x, body.y + (bot.entity.height || 1.8) * 0.55, body.z)
  const yaw = (opts.camYawDeg == null ? 35 : opts.camYawDeg) * Math.PI / 180
  const pitch = (opts.camPitchDeg == null ? 22 : opts.camPitchDeg) * Math.PI / 180
  const want = Math.max(1.5, Math.min(48, opts.distance == null ? 7 : opts.distance))
  const ox = Math.cos(pitch) * Math.sin(yaw)
  const oy = Math.sin(pitch)
  const oz = Math.cos(pitch) * Math.cos(yaw)
  let dist = want
  let from = new Vec3(target.x + ox * dist, target.y + oy * dist, target.z + oz * dist)
  if (typeof bot.blockAt === 'function') {
    while (dist > 1.5 && isSolid(bot.blockAt(from, false))) {
      dist -= 0.5
      from = new Vec3(target.x + ox * dist, target.y + oy * dist, target.z + oz * dist)
    }
  }
  return { from, forward: norm(new Vec3(target.x - from.x, target.y - from.y, target.z - from.z)), distance: dist }
}

/**
 * Render the world from either camera.
 * @param {object} bot Mineflayer bot
 * @param {object} opts `{ mode, width, height, maxDist, fovDeg, camYawDeg, camPitchDeg, distance }`
 */
function sampleWorldView (bot, opts = {}) {
  const width = opts.width || 192
  const height = opts.height || 108
  const maxDist = opts.maxDist || 48
  const mode = opts.mode === 'first' ? 'first' : 'orbit'
  const rgb = new Array(width * height * 3).fill(0)
  const meta = { w: width, h: height, rgb, mode, maxDist, botScreen: null }
  if (!bot || !bot.entity) return meta

  let from
  let forward
  let distance = 0
  if (mode === 'first') {
    from = bot.entity.position.offset(0, bot.entity.eyeHeight || 1.62, 0)
    forward = lookDir(bot.entity.yaw, bot.entity.pitch)
  } else {
    const cam = orbitEye(bot, opts)
    from = cam.from
    forward = cam.forward
    distance = cam.distance
    // The camera looks straight at the bot, so it is always dead centre.
    meta.botScreen = { x: 0.5, y: 0.5 }
  }
  meta.distance = distance
  meta.camYawDeg = opts.camYawDeg == null ? 35 : opts.camYawDeg
  meta.camPitchDeg = opts.camPitchDeg == null ? 22 : opts.camPitchDeg

  const worldUp = new Vec3(0, 1, 0)
  let right = cross(forward, worldUp)
  if (Math.hypot(right.x, right.y, right.z) < 1e-6) right = new Vec3(1, 0, 0)
  right = norm(right)
  const up = norm(cross(right, forward))
  const fov = (opts.fovDeg || 70) * Math.PI / 180
  const aspect = width / height
  const tanX = Math.tan(fov / 2)
  const tanY = tanX / aspect

  for (let j = 0; j < height; j++) {
    const sy = (0.5 - (j + 0.5) / height) * 2 * tanY
    for (let i = 0; i < width; i++) {
      const sx = ((i + 0.5) / width - 0.5) * 2 * tanX
      const dir = norm(new Vec3(
        forward.x + right.x * sx + up.x * sy,
        forward.y + right.y * sx + up.y * sy,
        forward.z + right.z * sx + up.z * sy,
      ))
      const hit = hitAlongRay(bot, from, dir, maxDist)
      const [r, g, b] = colorForBlock(hit && hit.name, hit ? hit.dist : maxDist, maxDist, hit)
      const o = (j * width + i) * 3
      rgb[o] = r
      rgb[o + 1] = g
      rgb[o + 2] = b
    }
  }
  return meta
}

/** Back-compat: the old first-person-only entry point. */
function sampleEyeView (bot, opts = {}) {
  return sampleWorldView(bot, { ...opts, mode: opts.mode || 'first' })
}

function packRgb (rgb) {
  return Buffer.from(Uint8Array.from(rgb)).toString('base64')
}

function packLum (lum) {
  const u8 = Uint8Array.from(lum, v => Math.max(0, Math.min(255, Math.round(Number(v) * 255))))
  return Buffer.from(u8).toString('base64')
}

function packEye (eye) {
  return {
    w: eye.w,
    h: eye.h,
    rgb: packRgb(eye.rgb),
    mode: eye.mode,
    botScreen: eye.botScreen,
    distance: eye.distance,
    camYawDeg: eye.camYawDeg,
    camPitchDeg: eye.camPitchDeg,
  }
}

module.exports = {
  sampleWorldView,
  sampleEyeView,
  colorForBlock,
  voxelHit,
  orbitEye,
  packRgb,
  packLum,
  packEye,
}

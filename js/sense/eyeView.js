/** First-person grid from the same world rays the fly visual cortex uses. */

const { Vec3 } = require('vec3')

function colorForBlock (name, dist) {
  const fog = Math.min(1, (dist || 0) / 28)
  let r = 92
  let g = 140
  let b = 210
  if (name) {
    const n = String(name).replace(/^minecraft:/, '')
    if (n.includes('oak_log') || n.includes('oak_wood')) { r = 120; g = 85; b = 48 }
    else if (n.includes('spruce')) { r = 78; g = 55; b = 36 }
    else if (n.includes('birch')) { r = 196; g = 176; b = 128 }
    else if (n.includes('cherry')) { r = 168; g = 92; b = 98 }
    else if (n.includes('acacia')) { r = 150; g = 78; b = 42 }
    else if (n.includes('dark_oak')) { r = 52; g = 36; b = 22 }
    else if (n.includes('jungle') || n.includes('mangrove')) { r = 92; g = 62; b = 32 }
    else if (n.includes('pale_oak')) { r = 186; g = 176; b = 164 }
    else if (n.includes('leaves')) { r = 46; g = 110; b = 42 }
    else if (n.includes('grass')) { r = 72; g = 130; b = 52 }
    else if (n.includes('dirt') || n.includes('podzol')) { r = 110; g = 78; b = 48 }
    else if (n.includes('sand')) { r = 210; g = 196; b = 130 }
    else if (n.includes('water') || n.includes('kelp')) { r = 40; g = 90; b = 180 }
    else if (n.includes('stone') || n.includes('cobble') || n.includes('ore')) { r = 110; g = 110; b = 118 }
    else if (n.includes('snow') || n === 'air') { r = 220; g = 228; b = 236 }
    else if (n.includes('log') || n.includes('wood')) { r = 100; g = 72; b = 40 }
    else { r = 90; g = 96; b = 72 }
    r = Math.round(r * (1 - 0.55 * fog) + 92 * 0.55 * fog)
    g = Math.round(g * (1 - 0.55 * fog) + 140 * 0.55 * fog)
    b = Math.round(b * (1 - 0.55 * fog) + 210 * 0.55 * fog)
  } else {
    const t = Math.max(0, Math.min(1, 0.35 + fog * 0.4))
    r = Math.round(70 + 90 * t)
    g = Math.round(120 + 70 * t)
    b = Math.round(190 + 40 * t)
  }
  return [r, g, b]
}

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
  while (t <= maxDist) {
    const b = bot.blockAt(new Vec3(x, y, z), false)
    if (b && b.name !== 'air' && b.boundingBox !== 'empty') {
      return { name: b.name, dist: t }
    }
    if (tmaxX < tmaxY) {
      if (tmaxX < tmaxZ) {
        t = tmaxX
        tmaxX += tdx
        x += sx
      } else {
        t = tmaxZ
        tmaxZ += tdz
        z += sz
      }
    } else if (tmaxY < tmaxZ) {
      t = tmaxY
      tmaxY += tdy
      y += sy
    } else {
      t = tmaxZ
      tmaxZ += tdz
      z += sz
    }
  }
  return null
}

function hitAlongRay (bot, from, dir, maxDist) {
  if (bot.world && typeof bot.world.raycast === 'function') {
    const hit = bot.world.raycast(from, dir, maxDist, blk => blk && blk.boundingBox !== 'empty')
    if (hit) {
      const pos = hit.position || from
      return { name: hit.name, dist: typeof from.distanceTo === 'function' ? from.distanceTo(pos) : maxDist }
    }
    return null
  }
  if (typeof bot.blockAt !== 'function') return null
  return voxelHit(bot, from, dir, maxDist)
}

function sampleEyeView (bot, opts = {}) {
  const width = opts.width || 64
  const height = opts.height || 36
  const maxDist = opts.maxDist || 24
  const rgb = new Array(width * height * 3).fill(0)
  if (!bot || !bot.entity) return { w: width, h: height, rgb }
  const yaw = bot.entity.yaw
  const pitch = bot.entity.pitch
  const eyeH = bot.entity.eyeHeight || 1.62
  const from = bot.entity.position.offset(0, eyeH, 0)
  const fovX = (opts.fovXDeg || 80) * Math.PI / 180
  const fovY = (opts.fovYDeg || 50) * Math.PI / 180

  for (let j = 0; j < height; j++) {
    const v = ((j + 0.5) / height - 0.5) * fovY
    for (let i = 0; i < width; i++) {
      const u = ((i + 0.5) / width - 0.5) * fovX
      const lookYaw = yaw - u
      const lookPitch = pitch + v
      const cp = Math.cos(lookPitch)
      const dir = new Vec3(-Math.sin(lookYaw) * cp, -Math.sin(lookPitch), -Math.cos(lookYaw) * cp)
      const hit = hitAlongRay(bot, from, dir, maxDist)
      const [r, g, b] = colorForBlock(hit && hit.name, hit ? hit.dist : maxDist)
      const o = (j * width + i) * 3
      rgb[o] = r
      rgb[o + 1] = g
      rgb[o + 2] = b
    }
  }
  return { w: width, h: height, rgb }
}

module.exports = { sampleEyeView, colorForBlock }

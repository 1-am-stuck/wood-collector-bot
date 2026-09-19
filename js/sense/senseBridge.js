const { emptyFrame, addOdor, addTaste, clampOdor } = require('./frame')

function stripNs (name) {
  if (!name) return ''
  return String(name).replace(/^minecraft:/, '')
}

function resolveTaste (table, name) {
  let row = table.contact[stripNs(name)]
  const seen = new Set()
  while (row && row.alias && !seen.has(row.alias)) {
    seen.add(row.alias)
    row = table.contact[row.alias]
  }
  return row || null
}

function forward (yawDeg) {
  const y = yawDeg * Math.PI / 180
  return { x: -Math.sin(y), y: 0, z: Math.cos(y) }
}

function right (yawDeg) {
  const y = yawDeg * Math.PI / 180
  return { x: -Math.cos(y), y: 0, z: -Math.sin(y) }
}

function toHead (dx, dy, dz, yawDeg) {
  const f = forward(yawDeg)
  const r = right(yawDeg)
  const len = Math.hypot(dx, dy, dz)
  if (len < 1e-9) return { az: 0, el: 0 }
  const xf = (dx * f.x + dz * f.z) / len
  const zr = (dx * r.x + dz * r.z) / len
  const yu = dy / len
  return {
    az: Math.atan2(zr, xf) * 180 / Math.PI,
    el: Math.asin(Math.max(-1, Math.min(1, yu))) * 180 / Math.PI,
  }
}

function wrapDeg (d) {
  while (d > 180) d -= 360
  while (d < -180) d += 360
  return d
}

function retinaDirections (retina) {
  const cols = retina.columns
  const [az0, az1] = retina.azimuthDeg
  const [el0, el1] = retina.elevationDeg
  const eyes = retina.eyes || ['L', 'R']
  const perEye = Math.floor(cols / eyes.length)
  const dirs = []
  for (let e = 0; e < eyes.length; e++) {
    const eyeSign = eyes[e] === 'L' ? -1 : 1
    for (let i = 0; i < perEye; i++) {
      const t = perEye === 1 ? 0.5 : i / (perEye - 1)
      const az = (az0 + (az1 - az0) * t) * (e === 0 && eyes.length === 2 ? 0.55 : 1)
      const el = el0 + (el1 - el0) * ((i * 3) % perEye) / Math.max(1, perEye - 1)
      const azRad = (az * eyeSign) * Math.PI / 180
      const elRad = el * Math.PI / 180
      dirs.push({
        eye: eyes[e],
        az: az * eyeSign,
        el,
        dx: Math.cos(elRad) * Math.cos(azRad),
        dy: Math.sin(elRad),
        dz: Math.cos(elRad) * Math.sin(azRad),
      })
    }
  }
  while (dirs.length < cols) dirs.push(dirs[dirs.length - 1] || { eye: 'R', az: 0, el: 0, dx: 1, dy: 0, dz: 0 })
  return dirs.slice(0, cols)
}

function albedoFor (name) {
  const n = stripNs(name)
  if (!n || n === 'air') return 0.5
  if (n.includes('log') || n.includes('wood')) return 0.25
  if (n.includes('leaves')) return 0.2
  if (n.includes('snow') || n.includes('quartz') || n === 'white_wool') return 0.9
  if (n.includes('obsidian') || n.includes('coal') || n.includes('black')) return 0.05
  if (n.includes('lava') || n === 'fire' || n.includes('torch')) return 0.7
  if (n.includes('water')) return 0.15
  if (n.includes('dirt') || n.includes('grass')) return 0.3
  if (n.includes('stone') || n.includes('cobble')) return 0.4
  return 0.35
}

function applyOdorSources (frame, sources, cfg, origin, yaw, radius, falloff, kind) {
  let bx = 0
  let bz = 0
  let total = 0
  for (const src of sources) {
    const name = stripNs(src.name)
    const aff = cfg.sources[name]
    if (!aff) continue
    const dx = src.x - origin.x
    const dy = src.y - origin.y
    const dz = src.z - origin.z
    const d = Math.hypot(dx, dy, dz)
    if (d > radius) continue
    const countScale = kind === 'item' ? (0.6 + 0.4 * Math.min(1, (src.count || 1) / 8)) : (kind === 'block' ? 0.5 : 1)
    const conc = Math.exp(-d / falloff) * countScale
    for (const [glom, w] of Object.entries(aff)) addOdor(frame, glom, w * conc)
    if (d > 1e-6) {
      bx += (dx / d) * conc
      bz += (dz / d) * conc
      total += conc
    }
  }
  return { bx, bz, total }
}

function objectChannelsFrom (objects, yawRate) {
  const ch = { LC4: 0, LPLC2: 0, LC11: 0, LC18: 0, LC10a: 0, LC15: 0, HS: 0 }
  for (const o of objects) {
    const exp = Math.max(0, o.expansionDegPerS)
    ch.LC4 = Math.max(ch.LC4, hillLike(exp / 200))
    const sizeGauss = Math.exp(-((o.angularSizeDeg - 60) ** 2) / (2 * 25 * 25))
    ch.LPLC2 = Math.max(ch.LPLC2, sizeGauss)
    if (o.angularSizeDeg < 15 && o.angularSpeedDegPerS > 5) {
      ch.LC11 = Math.max(ch.LC11, hillLike(o.angularSpeedDegPerS / 80))
      ch.LC18 = Math.max(ch.LC18, hillLike(o.angularSpeedDegPerS / 80) * 0.7)
    }
    if (o.flyLike && Math.abs(o.azimuthDeg) < 40) ch.LC10a = Math.max(ch.LC10a, 0.8)
    if (o.angularSizeDeg > 25 && Math.abs(o.elevationDeg) < 20) ch.LC15 = Math.max(ch.LC15, 0.4)
  }
  ch.HS = hillLike(Math.abs(yawRate) / 300)
  return ch
}

function hillLike (s) {
  if (s <= 0) return 0
  const sn = s ** 1.5
  return Math.min(1, sn / (0.2 ** 1.5 + sn))
}

function sampleWorld (world, cfg, state = {}) {
  const retina = cfg.retina
  const n = retina.columns
  const frame = emptyFrame(n)
  const dt = world.dtS || 0.05
  const yaw = world.yawDeg || 0
  const pos = world.position || { x: 0, y: 64, z: 0 }
  const vel = world.velocity || { x: 0, y: 0, z: 0 }

  if (state.prevYaw != null) {
    frame.yawRateDegPerS = wrapDeg(yaw - state.prevYaw) / dt
  }
  state.prevYaw = yaw
  if (state.prevPitch != null && world.pitchDeg != null) {
    frame.pitchRateDegPerS = (world.pitchDeg - state.prevPitch) / dt
  }
  state.prevPitch = world.pitchDeg || 0

  const speed = Math.hypot(vel.x, vel.z)
  const airspeed = Math.min(1, speed / Math.max(1e-3, cfg.mechano.flightSpeedBlocksPerS / 20))
  const r = right(yaw)
  const side = speed > 1e-4 ? (vel.x * r.x + vel.z * r.z) / speed : 0
  const lat = cfg.mechano.windLateralGain
  frame.windLeft = airspeed * (1 - lat * side)
  frame.windRight = airspeed * (1 + lat * side)
  frame.airborne = !world.onGround
  frame.legsOnGround = !!world.onGround
  frame.wingbeat = world.airborne || !world.onGround ? (world.wingbeat != null ? world.wingbeat : 0) : 0
  if (world.jumping) frame.wingbeat = Math.max(frame.wingbeat, 0.4)

  frame.damage = world.damageThisTick || 0
  if (world.horizontalCollision) {
    frame.touchHead = cfg.mechano.collisionTouch.head
    frame.touchLegs = cfg.mechano.collisionTouch.legs
  }
  if (world.raining) {
    frame.groomDust = cfg.mechano.rain.groomDust
    frame.moist = cfg.mechano.rain.moist
  }
  if (world.inWater) {
    frame.moist = Math.max(frame.moist, cfg.mechano.water.moist)
    frame.touchLegs = Math.max(frame.touchLegs, cfg.mechano.water.touchLegs)
  }
  const temp = world.temperature == null ? 0.8 : world.temperature
  if (temp > cfg.mechano.hotTempAbove) frame.hot = Math.min(1, temp - 1)
  if (temp < cfg.mechano.coldTempBelow) frame.cold = Math.min(1, (0.3 - temp) / 0.3)
  if (temp > cfg.mechano.dryTempAbove && !world.raining) frame.dry = 0.6

  if (world.soundHigh) frame.soundHigh = world.soundHigh
  if (world.soundLow) frame.soundLow = world.soundLow
  if (world.thunder) frame.soundHigh = Math.max(frame.soundHigh, cfg.mechano.thunderSoundHigh)
  if (world.explosion) frame.soundHigh = Math.max(frame.soundHigh, cfg.mechano.explosionSoundHigh)
  if (world.nearbyFootsteps) frame.soundLow = Math.max(frame.soundLow, cfg.mechano.footstepSoundLow)

  const itemR = cfg.itemOdor.radius
  const itemFall = cfg.itemOdor.falloffBlocks
  const blockR = cfg.blockOdor.blockRadius
  const blockFall = cfg.blockOdor.falloffBlocks
  const entR = cfg.entityOdor.radius
  const entFall = cfg.entityOdor.falloffBlocks

  let bx = 0
  let bz = 0
  let total = 0
  const a = applyOdorSources(frame, world.items || [], cfg.itemOdor, pos, yaw, itemR, itemFall, 'item')
  bx += a.bx; bz += a.bz; total += a.total
  const b = applyOdorSources(frame, world.blocks || [], cfg.blockOdor, pos, yaw, blockR, blockFall, 'block')
  bx += b.bx; bz += b.bz; total += b.total
  const c = applyOdorSources(frame, (world.entities || []).map(e => ({ name: e.type || e.name, x: e.x, y: e.y, z: e.z })), cfg.entityOdor, pos, yaw, entR, entFall, 'entity')
  bx += c.bx; bz += c.bz; total += c.total
  if (total > 1e-6) {
    const h = toHead(bx, 0, bz, yaw)
    frame.odorBearingDeg = h.az
  }
  clampOdor(frame)

  const tasteName = world.standingOn || world.contactBlock || (world.heldItem && world.proboscisOut ? world.heldItem : null)
  const labellar = !!world.proboscisOut || !!world.inWater
  if (tasteName) {
    const row = resolveTaste(cfg.taste, tasteName)
    if (row) {
      for (const [grn, w] of Object.entries(row.tarsal || {})) addTaste(frame, grn, w)
      if (labellar) {
        for (const [grn, w] of Object.entries(row.labellar || {})) addTaste(frame, grn, w)
      }
    }
  }
  if (world.inWater || (world.raining && world.onGround)) {
    const water = resolveTaste(cfg.taste, 'water')
    if (water) {
      for (const [grn, w] of Object.entries(water.tarsal || {})) addTaste(frame, grn, w)
      // Rain on the body is a wet-contact event in Minecraft (no antenna hygrosensors).
      if (labellar || world.inWater || world.raining) {
        for (const [grn, w] of Object.entries(water.labellar || {})) addTaste(frame, grn, w)
      }
    }
  }

  state.prevObjects = state.prevObjects || {}
  const now = world.tick == null ? (state.tick = (state.tick || 0) + 1) : world.tick
  for (const e of world.entities || []) {
    const dx = e.x - pos.x
    const dy = e.y - pos.y
    const dz = e.z - pos.z
    const d = Math.hypot(dx, dy, dz)
    if (d < 0.05) continue
    const radius = 0.5 * Math.max(e.w || 0.6, e.h || 1.8, e.d || 0.6)
    const size = 2 * Math.atan(radius / d) * 180 / Math.PI
    const head = toHead(dx, dy, dz, yaw)
    const prev = state.prevObjects[e.id]
    let expansion = 0
    let angSpeed = 0
    if (prev && now - prev.tick <= 2) {
      const ticks = Math.max(1, now - prev.tick)
      expansion = (size - prev.size) / (ticks * dt)
      const daz = wrapDeg(head.az - prev.az)
      const del = head.el - prev.el
      angSpeed = Math.hypot(daz, del) / (ticks * dt)
    }
    state.prevObjects[e.id] = { size, az: head.az, el: head.el, tick: now }
    const type = e.type || e.name || ''
    frame.objects.push({
      azimuthDeg: head.az,
      elevationDeg: head.el,
      angularSizeDeg: size,
      expansionDegPerS: expansion,
      angularSpeedDegPerS: angSpeed,
      contrast: type.includes('fly') ? 0.9 : 0.8,
      flyLike: type.includes('fly') || type === 'player',
      name: type,
    })
  }

  const dirs = retinaDirections(retina)
  const rayLen = retina.maxDistance
  const dayFactor = world.dayFactor == null ? 1 : world.dayFactor
  const blocks = world.blocks || []
  for (let i = 0; i < n; i++) {
    const dir = dirs[i]
    const f = forward(yaw)
    const rr = right(yaw)
    const wx = f.x * dir.dx + rr.x * dir.dz
    const wy = dir.dy
    const wz = f.z * dir.dx + rr.z * dir.dz
    let hit = null
    let bestT = rayLen
    for (const blk of blocks) {
      const bx2 = blk.x + 0.5 - pos.x
      const by2 = blk.y + 0.5 - pos.y
      const bz2 = blk.z + 0.5 - pos.z
      const t = bx2 * wx + by2 * wy + bz2 * wz
      if (t <= 0 || t > bestT) continue
      const px = pos.x + wx * t
      const py = pos.y + wy * t
      const pz = pos.z + wz * t
      if (Math.abs(px - (blk.x + 0.5)) < 0.6 && Math.abs(py - (blk.y + 0.5)) < 0.6 && Math.abs(pz - (blk.z + 0.5)) < 0.6) {
        hit = blk
        bestT = t
      }
    }
    if (!hit) {
      frame.luminance[i] = wy > -0.2 ? 0.85 * dayFactor + 0.05 : 0.25
    } else {
      const light = (hit.light == null ? 12 : hit.light) / 15
      const alb = hit.albedo == null ? albedoFor(hit.name) : hit.albedo
      frame.luminance[i] = light * dayFactor * (0.3 + 0.7 * alb)
    }
  }
  for (const o of frame.objects) {
    if (o.angularSizeDeg < 2) continue
    for (let i = 0; i < n; i++) {
      const dir = dirs[i]
      if (Math.hypot(wrapDeg(dir.az - o.azimuthDeg), dir.el - o.elevationDeg) < o.angularSizeDeg / 2) {
        const cur = frame.luminance[i]
        frame.luminance[i] = Number.isNaN(cur) ? 0.1 : Math.min(cur, 0.1 + 0.3 * (1 - o.contrast))
      }
    }
  }

  frame.objectChannels = objectChannelsFrom(frame.objects, frame.yawRateDegPerS)
  return frame
}

/**
 * Sample a live Mineflayer bot into a World snapshot, then SenseBridge.
 */
function sampleBot (bot, cfg, state = {}) {
  const p = bot.entity.position
  const yawDeg = (bot.entity.yaw * 180) / Math.PI
  const pitchDeg = (bot.entity.pitch * 180) / Math.PI
  const v = bot.entity.velocity
  const blocks = []
  const origin = p.floored()
  const r = cfg.blockOdor.blockRadius
  for (let dx = -r; dx <= r; dx++) {
    for (let dy = -3; dy <= 3; dy++) {
      for (let dz = -r; dz <= r; dz++) {
        const b = bot.blockAt(origin.offset(dx, dy, dz))
        if (!b || b.boundingBox === 'empty') continue
        const light = typeof b.light === 'number' ? b.light : (b.skyLight || 10)
        blocks.push({ name: b.name, x: b.position.x, y: b.position.y, z: b.position.z, light })
      }
    }
  }
  const items = []
  const entities = []
  for (const e of Object.values(bot.entities)) {
    if (!e || !e.position) continue
    if (e === bot.entity) continue
    if (e.name === 'item' || e.displayName === 'Item') {
      const iname = e.metadata && e.getDroppedItem ? (e.getDroppedItem() || {}).name : e.displayName
      items.push({
        name: iname || 'apple',
        x: e.position.x,
        y: e.position.y,
        z: e.position.z,
        count: 1,
      })
    } else {
      entities.push({
        id: e.id,
        type: e.username ? 'player' : (e.name || e.kind || 'unknown'),
        x: e.position.x,
        y: e.position.y,
        z: e.position.z,
        w: e.width || 0.6,
        h: e.height || 1.8,
        d: e.width || 0.6,
      })
    }
  }
  const below = bot.blockAt(origin.offset(0, -1, 0))
  const held = bot.heldItem ? bot.heldItem.name : null
  const world = {
    tick: bot.time ? bot.time.age : undefined,
    dtS: 0.05,
    position: { x: p.x, y: p.y, z: p.z },
    yawDeg,
    pitchDeg,
    velocity: { x: v.x, y: v.y, z: v.z },
    onGround: !!bot.entity.onGround,
    inWater: typeof bot.entity.isInWater === 'function' ? bot.entity.isInWater() : false,
    raining: !!(bot.isRaining),
    temperature: 0.8,
    damageThisTick: 0,
    horizontalCollision: !!bot.entity.isCollidedHorizontally,
    standingOn: below ? below.name : null,
    heldItem: held,
    proboscisOut: false,
    dayFactor: bot.time ? 1 - (bot.time.skyLightSubtracted || 0) / 15 : 1,
    blocks,
    items,
    entities,
  }
  return sampleWorld(world, cfg, state)
}

module.exports = {
  sampleWorld,
  sampleBot,
  retinaDirections,
  toHead,
  forward,
  right,
  stripNs,
  resolveTaste,
  albedoFor,
}

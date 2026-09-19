/** Descending commands -> a fly that can take off, cruise, and land.
 *
 * This is a fruit fly. Mineflayer's player is the body. DNp01 (jump) is
 * takeoff: gravity off, body climbs. While airborne, DNp09/MDN translate
 * along the gaze -- walking keys do nothing with gravity off, so the pose
 * has to move. DNg60 (noop) is halt: the wings stop, gravity comes back,
 * and the legs can walk. A later DNp01 spike takes off again.
 *
 * At 15 Hz a step is ~67 ms, so an action cannot block. Locomotion is a
 * held state. A turn steers without cancelling it. Nothing here
 * second-guesses the policy.
 */

const ACTIONS = ['forward', 'back', 'turn_left', 'turn_right', 'jump', 'mine', 'camera_up', 'camera_down', 'noop']

const HALF_PI = Math.PI / 2
const motorState = new WeakMap()

function actionIndex (name) {
  const i = ACTIONS.indexOf(name)
  return i < 0 ? ACTIONS.indexOf('noop') : i
}

function isCreative (bot) {
  if (!bot || !bot.game) return false
  const mode = bot.game.gameMode
  return mode === 'creative' || mode === 1
}

function withTimeout (p, ms) {
  let t
  return Promise.race([
    Promise.resolve(p).finally(() => clearTimeout(t)),
    new Promise(resolve => { t = setTimeout(resolve, ms) }),
  ])
}

function motorFor (bot) {
  let state = motorState.get(bot)
  if (!state) {
    state = { locomotion: 'stop', digging: false, airborne: false }
    motorState.set(bot, state)
  }
  return state
}

function setControl (bot, key, on) {
  try { bot.setControlState(key, on) } catch (_) {}
}

/** Start a dig without waiting for it; digging spans many ticks. */
function beginDig (bot, state, reach) {
  if (state.digging) return
  const target = typeof bot.blockAtCursor === 'function' ? bot.blockAtCursor(reach) : null
  if (!target) return
  state.digging = true
  Promise.resolve()
    .then(() => bot.dig(target))
    .catch(() => {})
    .then(() => { state.digging = false })
}

function endDig (bot, state) {
  if (!state.digging) return
  state.digging = false
  try { bot.stopDigging() } catch (_) {}
}

/**
 * Apply one descending command.
 * @param {object} bot Mineflayer bot
 * @param {string} name one of ACTIONS
 * @param {object} [control] `configs/action_space.json` control block
 */
function keepFlying (bot) {
  if (!bot) return
  motorFor(bot).airborne = true
  if (bot.creative && typeof bot.creative.startFlying === 'function') {
    try { bot.creative.startFlying() } catch (_) {}
  }
}

/** Halt the wings. Gravity returns so the body can perch and walk. */
async function land (bot) {
  if (!bot) return
  const state = motorFor(bot)
  state.airborne = false
  state.locomotion = 'stop'
  for (const key of ['forward', 'back', 'jump', 'sprint']) setControl(bot, key, false)
  if (bot.creative && typeof bot.creative.stopFlying === 'function') {
    try { await withTimeout(bot.creative.stopFlying(), 400) } catch (_) {}
  }
}

/** Gaze-direction unit vector in Mineflayer yaw/pitch (0 yaw = +Z, up is −pitch). */
function gazeVector (yaw, pitch) {
  const cp = Math.cos(pitch)
  return {
    x: -Math.sin(yaw) * cp,
    y: -Math.sin(pitch),
    z: -Math.cos(yaw) * cp,
  }
}

function translate (bot, dx, dy, dz) {
  const p = bot.entity && bot.entity.position
  if (!p) return
  p.x += dx
  p.y += dy
  p.z += dz
}

/**
 * Move the airborne body. Walking keys do nothing useful with gravity off, so
 * the descending command has to change the pose itself.
 */
function applyFlight (bot, state, name, control = {}) {
  if (!state.airborne) return
  const dt = Math.max(0.02, (control.dtMs || 67) / 1000)
  const speed = (control.flySpeed || 8) * dt
  const yaw = bot.entity.yaw || 0
  const pitch = bot.entity.pitch || 0
  let dx = 0
  let dy = 0
  let dz = 0
  if (state.locomotion === 'forward' || state.locomotion === 'back') {
    const g = gazeVector(yaw, pitch)
    const s = state.locomotion === 'back' ? -speed : speed
    dx += g.x * s
    dy += g.y * s
    dz += g.z * s
  }
  if (name === 'jump') dy += speed
  if (dx === 0 && dy === 0 && dz === 0) return
  translate(bot, dx, dy, dz)
}

async function applyAction (bot, name, control = {}) {
  if (!bot || !bot.entity) return
  const state = motorFor(bot)
  const turn = (control.turnDeg || 12) * Math.PI / 180
  const pitchStep = (control.pitchDeg || 8) * Math.PI / 180
  const reach = control.reachBlocks || 5
  const lookMs = control.lookTimeoutMs || 40

  if (name === 'forward') state.locomotion = 'forward'
  else if (name === 'back') state.locomotion = 'back'
  else if (name === 'noop') {
    await land(bot)
    return
  }

  if (name === 'jump') keepFlying(bot)

  if (name === 'mine') beginDig(bot, state, reach)
  else endDig(bot, state)

  setControl(bot, 'forward', state.locomotion === 'forward')
  setControl(bot, 'back', state.locomotion === 'back')
  setControl(bot, 'sprint', state.locomotion === 'forward' && !!control.sprint)
  // Jump is momentary: held only on the tick the command arrives.
  setControl(bot, 'jump', name === 'jump')

  let yaw = bot.entity.yaw
  let pitch = bot.entity.pitch
  if (name === 'turn_left') yaw += turn
  else if (name === 'turn_right') yaw -= turn
  else if (name === 'camera_up') pitch = Math.max(-HALF_PI, pitch - pitchStep)
  else if (name === 'camera_down') pitch = Math.min(HALF_PI, pitch + pitchStep)

  if (name === 'turn_left' || name === 'turn_right' || name === 'camera_up' || name === 'camera_down') {
    try { await withTimeout(bot.look(yaw, pitch, true), lookMs) } catch (_) {}
  }

  applyFlight(bot, state, name, control)
}

/** Release everything, e.g. between episodes. */
function releaseControls (bot) {
  if (!bot) return
  const state = motorFor(bot)
  state.locomotion = 'stop'
  endDig(bot, state)
  for (const key of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
    setControl(bot, key, false)
  }
}

/** Creative reposition, used only when an episode resets. */
async function flyOffset (bot, dx, dy, dz, timeoutMs = 500) {
  if (!bot || !bot.creative || !bot.entity) return false
  try { bot.creative.startFlying() } catch (_) {}
  const dest = bot.entity.position.offset(dx, dy, dz)
  try {
    await withTimeout(bot.creative.flyTo(dest), timeoutMs)
    return true
  } catch (_) {
    return false
  }
}

function expertActionFromIntent (intent) {
  if (!intent) return 'noop'
  if (intent === 'approach' || intent === 'explore' || intent === 'forward') return 'forward'
  if (intent === 'collect' || intent === 'mine') return 'mine'
  if (intent === 'left' || intent === 'turn_left') return 'turn_left'
  if (intent === 'right' || intent === 'turn_right') return 'turn_right'
  if (intent === 'jump') return 'jump'
  return 'noop'
}

module.exports = {
  ACTIONS,
  actionIndex,
  applyAction,
  keepFlying,
  land,
  applyFlight,
  releaseControls,
  expertActionFromIntent,
  withTimeout,
  flyOffset,
  isCreative,
  motorFor,
}

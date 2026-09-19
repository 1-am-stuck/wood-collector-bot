const ACTIONS = ['forward', 'back', 'turn_left', 'turn_right', 'jump', 'mine', 'camera_up', 'camera_down', 'noop']

function actionIndex (name) {
  const i = ACTIONS.indexOf(name)
  return i < 0 ? ACTIONS.indexOf('noop') : i
}

async function applyAction (bot, name, control = {}) {
  const turn = (control.turnDeg || 15) * Math.PI / 180
  const pitch = (control.pitchDeg || 10) * Math.PI / 180
  const hold = control.forwardTicks || 4
  switch (name) {
    case 'forward':
      bot.setControlState('forward', true)
      await sleep(hold * 50)
      bot.setControlState('forward', false)
      break
    case 'back':
      bot.setControlState('back', true)
      await sleep(hold * 50)
      bot.setControlState('back', false)
      break
    case 'turn_left':
      await bot.look(bot.entity.yaw + turn, bot.entity.pitch, true)
      break
    case 'turn_right':
      await bot.look(bot.entity.yaw - turn, bot.entity.pitch, true)
      break
    case 'jump':
      bot.setControlState('jump', true)
      await sleep(200)
      bot.setControlState('jump', false)
      break
    case 'mine': {
      const target = bot.blockAtCursor && bot.blockAtCursor(5)
      if (target) {
        try { await bot.dig(target) } catch (_) {}
      }
      break
    }
    case 'camera_up':
      await bot.look(bot.entity.yaw, Math.max(-Math.PI / 2, bot.entity.pitch - pitch), true)
      break
    case 'camera_down':
      await bot.look(bot.entity.yaw, Math.min(Math.PI / 2, bot.entity.pitch + pitch), true)
      break
    default:
      await sleep(50)
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

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

module.exports = { ACTIONS, actionIndex, applyAction, expertActionFromIntent }

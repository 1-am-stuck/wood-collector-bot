const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const {
  missingWoodTypes,
  nextAction,
  isOscillating,
  nearestUnblacklisted,
  blacklistTree,
  rememberHere,
  pickExploreGoal,
  posKey,
  mapLocal,
  bestStepUp,
} = require('./brain')

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'Lumberjack',
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(require('mineflayer-collectblock').plugin)

let mcData
let busy = false
const blacklist = new Set()
const trail = []
const recentPathLens = []
const recentHopKeys = new Set()

bot.once('spawn', () => {
  mcData = require('minecraft-data')(bot.version)
  bot.pathfinder.setMovements(new Movements(bot))

  // Needs `op Lumberjack` in the server console once
  bot.chat('/give @s diamond_axe')
  bot.chat('/give @s cobblestone 192')

  const p = bot.entity.position.floored()
  rememberHere(trail, p)
  bot.chat(`/setblock ${p.x + 2} ${p.y} ${p.z + 2} minecraft:chest`)
})

bot.on('chat', (username, message) => {
  if (username === bot.username) return
  if (message === 'go') collectAllWood()
})

bot.on('path_update', (r) => {
  const len = r.path ? r.path.length : 0
  console.log('path:', r.status, '-', len, 'moves')
  recentPathLens.push(len)
  if (recentPathLens.length > 8) recentPathLens.shift()
  if (isOscillating(recentPathLens)) {
    console.log('oscillating — stopping pathfinder')
    recentPathLens.length = 0
    bot.pathfinder.stop()
  }
})

async function collectAllWood () {
  if (busy) return
  busy = true
  try {
    while (true) {
      const missing = missingWoodTypes(bot.inventory.items().map(item => item.name))
      console.log('missing:', missing.join(', ') || '(none)')

      const target = missing.length ? findNearestMissingLog(missing) : null
      const targetDist = target ? bot.entity.position.distanceTo(target.position) : null
      const action = nextAction({ missing, nearbyTarget: target, targetDist })
      const toward = target ? target.position : null

      if (action.type === 'done') {
        bot.chat('I have every wood type!')
        return
      }

      if (await escapeHole(toward)) continue

      if (action.type === 'approach') {
        console.log('approach', target.name, 'at', target.position, 'dist', targetDist.toFixed(1))
        rememberHere(trail, bot.entity.position.floored())
        try {
          await bot.pathfinder.goto(new goals.GoalNear(
            target.position.x,
            target.position.y,
            target.position.z,
            3,
          ))
          rememberHere(trail, bot.entity.position.floored())
        } catch (err) {
          console.log('approach failed:', err.message)
          blacklistTree(blacklist, target.position)
        }
        continue
      }

      if (action.type === 'collect') {
        console.log('collect', target.name, 'at', target.position)
        try {
          await bot.collectBlock.collect(target)
          rememberHere(trail, bot.entity.position.floored())
          blacklistTree(blacklist, target.position)
        } catch (err) {
          console.log('collect failed:', err.message, '- skipping that tree')
          blacklistTree(blacklist, target.position)
        }
        continue
      }

      console.log('nothing nearby, exploring...')
      await explore(toward)
    }
  } finally {
    busy = false
  }
}

function findNearestMissingLog (missing) {
  const matching = missing
    .map(type => mcData.blocksByName[`${type}_log`])
    .filter(Boolean)
    .map(block => block.id)

  if (matching.length === 0) return null

  const positions = bot.findBlocks({
    matching,
    maxDistance: 64,
    count: 10,
  })

  const pos = nearestUnblacklisted(positions, blacklist, bot.entity.position)
  return pos ? bot.blockAt(pos) : null
}

async function escapeHole (toward) {
  const map = mapLocal(bot)
  if (!map.inHole) return false

  const step = bestStepUp(map, bot.entity.position, toward, { mustFace: false })
  const exit = step || map.holeExit
  if (!exit) return false

  const from = bot.entity.position.floored()
  const dest = from.offset(exit.dx, 1, exit.dz)
  console.log('in hole, stepping', exit.name)
  rememberHere(trail, from)
  try {
    await bot.pathfinder.goto(new goals.GoalNear(dest.x, dest.y, dest.z, 1))
    rememberHere(trail, bot.entity.position.floored())
  } catch (err) {
    console.log('hole escape failed:', err.message)
  }
  return true
}

async function explore (toward) {
  rememberHere(trail, bot.entity.position.floored())
  const recentKeys = new Set([
    ...trail.map(p => posKey(p)),
    ...recentHopKeys,
  ])
  const dest = pickExploreGoal(bot, bot.entity.position, toward, recentKeys, null)

  if (dest) {
    recentHopKeys.add(posKey(dest))
    if (recentHopKeys.size > 20) {
      const first = recentHopKeys.values().next().value
      recentHopKeys.delete(first)
    }
    console.log('explore hop →', dest.toString(), '| trail', trail.length)
    try {
      await bot.pathfinder.goto(new goals.GoalNear(dest.x, dest.y, dest.z, 1))
      rememberHere(trail, bot.entity.position.floored())
      return
    } catch (err) {
      console.log('explore hop failed:', err.message, '- backtracking')
    }
  } else {
    console.log('no hop, backtracking')
  }

  await backtrack()
}

async function backtrack () {
  if (trail.length === 0) {
    console.log('no trail left; waiting for a new hop next loop')
    return
  }

  const back = trail.pop()
  console.log('backtrack →', back.toString(), '| trail', trail.length)
  try {
    await bot.pathfinder.goto(new goals.GoalNear(back.x, back.y, back.z, 2))
  } catch (err) {
    console.log('backtrack failed:', err.message)
  }
}

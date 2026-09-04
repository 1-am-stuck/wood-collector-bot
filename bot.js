const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const { Vec3 } = require('vec3')

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'Lumberjack',
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(require('mineflayer-collectblock').plugin)

let mcData
const blacklist = new Set()

// Scratchpad: LIFO trail for backtracking, FIFO frontier for where to explore next
const trail = []
const frontier = []
const TRAIL_MAX = 20
const FRONTIER_BATCH = 4
const EXPLORE_MIN = 40
const EXPLORE_MAX = 80

bot.once('spawn', () => {
  mcData = require('minecraft-data')(bot.version)
  bot.pathfinder.setMovements(new Movements(bot))

  // Needs `op Lumberjack` in the server console once
  bot.chat('/give @s diamond_axe')
  bot.chat('/give @s cobblestone 192')

  const p = bot.entity.position.floored()
  rememberHere()
  bot.chat(`/setblock ${p.x + 2} ${p.y} ${p.z + 2} minecraft:chest`)
})

bot.on('chat', (username, message) => {
  if (username === bot.username) return
  if (message === 'go') collectAllWood()
})

bot.on('path_update', (r) => {
  console.log('path:', r.status, '-', r.path.length, 'moves')
})

const WOOD_TYPES = [
  'oak', 'spruce', 'birch', 'jungle', 'acacia',
  'dark_oak', 'mangrove', 'cherry', 'pale_oak',
]

async function collectAllWood () {
  while (true) {
    const missing = WOOD_TYPES.filter(type => !hasLog(type))
    console.log('missing:', missing.join(', ') || '(none)')

    if (missing.length === 0) {
      bot.chat('I have every wood type!')
      return
    }

    const target = findNearestMissingLog(missing)
    if (target) {
      console.log('found', target.name, 'at', target.position)
      try {
        await bot.collectBlock.collect(target)
        rememberHere()
      } catch (err) {
        console.log('collect failed:', err.message, '- skipping that tree')
        blacklist.add(target.position.toString())
      }
    } else {
      console.log('nothing nearby, exploring...')
      await explore()
    }
  }
}

function hasLog (type) {
  return bot.inventory.items().some(item => item.name === `${type}_log`)
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

  const pos = positions.find(p => !blacklist.has(p.toString()))
  return pos ? bot.blockAt(pos) : null
}

function rememberHere () {
  const p = bot.entity.position.floored()
  const last = trail[trail.length - 1]
  if (last && last.equals(p)) return
  trail.push(p)
  if (trail.length > TRAIL_MAX) trail.shift()
}

function refillFrontier () {
  const here = bot.entity.position
  for (let i = 0; i < FRONTIER_BATCH; i++) {
    const angle = Math.random() * Math.PI * 2
    const dist = EXPLORE_MIN + Math.random() * (EXPLORE_MAX - EXPLORE_MIN)
    frontier.push(new Vec3(
      Math.floor(here.x + Math.cos(angle) * dist),
      Math.floor(here.y),
      Math.floor(here.z + Math.sin(angle) * dist),
    ))
  }
}

async function explore () {
  if (frontier.length === 0) refillFrontier()

  const goal = frontier.shift()
  rememberHere()
  console.log('explore →', goal.toString(), '| trail', trail.length, '| frontier', frontier.length)

  try {
    await bot.pathfinder.goto(new goals.GoalXZ(goal.x, goal.z))
    rememberHere()
  } catch (err) {
    console.log('explore failed:', err.message, '- backtracking')
    await backtrack()
  }
}

async function backtrack () {
  if (trail.length === 0) {
    console.log('no trail left; inventing new frontier')
    refillFrontier()
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

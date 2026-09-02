const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'Lumberjack',
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(require('mineflayer-collectblock').plugin)

let mcData

//set up pathfinder
bot.once('spawn', () => {
  mcData = require('minecraft-data')(bot.version)
  bot.pathfinder.setMovements(new Movements(bot))
})

//wait for user to type'go' and start collecting wood
bot.on('chat', (username, message) => {
    if (username === bot.username) return
    if (message === 'go') collectAllWood()
  })

//list of wood types to collect
const WOOD_TYPES = ['oak', 'spruce', 'birch', 'jungle', 'acacia', 'dark_oak', 'mangrove', 'cherry', 'pale_oak']

async function collectAllWood () {
    while (true) {
      const missing = WOOD_TYPES.filter(type => !hasLog(type))
      if (missing.length === 0) {
        bot.chat('I have every wood type!')
        return
      }
  
      const target = findNearestMissingLog(missing)   // a block, or null
      if (target) {
        await bot.collectBlock.collect(target)
      } else {
        await explore()
      }
    }
  }

function hasLog(type) {
    return bot.inventory.items().some(item => item.name === type + '_log')
}

function findNearestMissingLog(missing) {
    return bot.findBlock({
        matching: missing.map(type => mcData.blocksByName[type + '_log'].id),
        maxDistance: 64,
        maxResults: 1
    })
}

//pick a random direction and go there, if the block is not found pick a new direction
async function explore() {
    const angle = Math.random() * Math.PI * 2
    const x = bot.entity.position.x + Math.cos(angle) * 100
    const z = bot.entity.position.z + Math.sin(angle) * 100
    try {
      await bot.pathfinder.goto(new goals.GoalXZ(x, z))
    } catch (err) {
      // unreachable spot — fine, the loop will pick a new direction
    }
  }
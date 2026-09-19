/**
 * Fruit fly in Minecraft: spawn, sense, walk around, interact.
 * Chat: stop | explore | log
 * GoalToSense is generic. Rarest log only selects which existing GoalSpec to inject.
 */
const path = require('path')
const mineflayer = require('mineflayer')
const { loadSenseConfig } = require('./js/sense/loadConfig')
const { startExplore } = require('./js/sense/exploreLoop')

const host = process.env.MC_HOST || '127.0.0.1'
const port = Number(process.env.MC_PORT || 25565)
const username = process.env.MC_USER || 'FruitFly'

const bot = mineflayer.createBot({ host, port, username })
const cfg = loadSenseConfig(path.join(__dirname))
let explore = null

bot.once('spawn', () => {
  console.log('spawned in', bot.game.dimension, 'at', bot.entity.position)
  explore = startExplore(bot, cfg, { exploreRadius: 32 })
  explore.run().catch(err => console.error('explore:', err))
})

bot.on('kicked', r => console.error('kicked', r))
bot.on('error', err => {
  console.error('minecraft:', err.message)
  if (err.code === 'ECONNREFUSED') {
    console.error('Paper is not listening on', host + ':' + port, '— start it (./start-server.sh) and rerun: node fly.js')
  }
})
bot.on('end', () => {
  if (explore) explore.stop()
  console.log('disconnected')
})

bot.on('chat', (who, message) => {
  if (who === bot.username) return
  if (message === 'stop' && explore) {
    explore.stop()
    bot.chat('stopped')
  }
  if (message === 'explore' && explore) {
    explore.run().catch(err => console.error(err))
  }
  if (message === 'log' && explore) {
    bot.chat('logging ' + explore.openLog(path.join(__dirname, 'logs', `explore_${Date.now()}.jsonl`)))
  }
})

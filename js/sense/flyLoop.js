const path = require('path')
const { loadSenseConfig, loadGoalSpec } = require('./loadConfig')
const { sampleBot } = require('./senseBridge')
const { applyGoal } = require('./goalToSense')
const { openJsonl, compactFrame } = require('./logger')
const { applyAction, expertActionFromIntent } = require('./actions')
const { connectPolicy } = require('./policyClient')

function startFlyRuntime (bot, opts = {}) {
  const root = opts.root || path.join(__dirname, '..', '..')
  const cfg = loadSenseConfig(root)
  const state = {}
  let goal = opts.goal ? loadGoalSpec(opts.goal, root) : null
  let log = null
  let client = null
  let running = false

  function setGoal (id) {
    goal = id ? loadGoalSpec(id, root) : null
    bot.chat(goal ? `goal ${goal.id}` : 'goal cleared')
  }

  function openLog (file) {
    log = openJsonl(file || path.join(root, 'logs', `sense_${Date.now()}.jsonl`))
    return log.path
  }

  function sample (worldOverride) {
    const worldFrame = sampleBot(bot, cfg, state)
    const world = worldOverride || null
    return goal ? applyGoal(worldFrame, goal, world) : worldFrame
  }

  async function connect (host, port) {
    client = connectPolicy(host || '127.0.0.1', port || 8765)
    const pong = await client.ping()
    return pong
  }

  async function runPolicy (maxSteps = 200) {
    if (running) return
    if (!client) await connect()
    running = true
    try {
      for (let i = 0; i < maxSteps; i++) {
        const frame = sample()
        const res = await client.act(frame, { greedy: true, goal_id: goal && goal.id })
        if (log) {
          log.write({
            t: Date.now(),
            goal_id: goal && goal.id,
            frame: compactFrame(frame),
            action: res.action,
            source: 'policy',
          })
        }
        await applyAction(bot, res.action, cfg.actions.control)
      }
    } finally {
      running = false
    }
  }

  function logExpert (intent, extra = {}) {
    if (!log) return
    const frame = sample()
    log.write({
      t: Date.now(),
      goal_id: goal && goal.id,
      frame: compactFrame(frame),
      action: expertActionFromIntent(intent),
      intent,
      source: 'expert',
      ...extra,
    })
  }

  return { cfg, setGoal, openLog, sample, connect, runPolicy, logExpert, getGoal: () => goal }
}

module.exports = { startFlyRuntime }

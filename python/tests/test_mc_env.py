from pathlib import Path


def test_minecraft_env_speaks_jsonl(tmp_path):
    script = tmp_path / "fake_rollout.js"
    script.write_text(
        """
console.log('injected env tip that is not json')
const readline = require('readline')
const rl = readline.createInterface({ input: process.stdin })
rl.on('line', line => {
  const msg = JSON.parse(line)
  if (msg.cmd === 'load_catalog') {
    console.log(JSON.stringify({ ok: true, logs: Object.keys(msg.catalog || {}) }))
  } else if (msg.cmd === 'connect') {
    console.log(JSON.stringify({ ok: true, username: 'FruitFly' }))
  } else if (msg.cmd === 'reset') {
    console.log(JSON.stringify({
      frame: { luminance: [0], odor: {}, taste: {}, objectChannels: {} },
      facts: { rarest: 'cherry_log', inventory: {}, distance: 10 },
      reward: 0, done: false,
    }))
  } else if (msg.cmd === 'step') {
    console.log(JSON.stringify({
      frame: { luminance: [0], odor: {}, taste: {}, objectChannels: {} },
      facts: { rarest: 'cherry_log', inventory: {}, distance: 9 },
      reward: 0.3, done: false, action: msg.action,
    }))
  } else if (msg.cmd === 'close') {
    console.log(JSON.stringify({ ok: true }))
    process.exit(0)
  } else {
    console.log(JSON.stringify({ error: 'unknown' }))
  }
})
""",
        encoding="utf-8",
    )
    from fly_policy.mc_env import MinecraftEnv

    env = MinecraftEnv(
        {
            "minecraft": {"host": "127.0.0.1", "port": 25565, "username": "FruitFly"},
            "goal": {"catalog": {"cherry_log": {"id": "collect:cherry_log"}}, "explore_radius": 32},
        },
        node_script=script,
    )
    env.start()
    obs = env.reset()
    assert obs["facts"]["rarest"] == "cherry_log"
    step = env.step("forward")
    assert step["reward"] == 0.3
    assert step["action"] == "forward"
    env.close()



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


def test_minecraft_env_passes_feed_mode_and_goalspec(tmp_path):
    script = tmp_path / "fake_rollout.js"
    script.write_text(
        """
const readline = require('readline')
const rl = readline.createInterface({ input: process.stdin })
rl.on('line', line => {
  const msg = JSON.parse(line)
  if (msg.cmd === 'load_catalog') {
    console.log(JSON.stringify({ ok: true, logs: [] }))
  } else if (msg.cmd === 'connect') {
    console.log(JSON.stringify({
      ok: true,
      username: 'FruitFly',
      mode: msg.mode,
      seed_rich: msg.seed_rich,
      census: msg.census,
      goal_id: (msg.goal_spec || {}).id,
      success: (msg.goal_spec || {}).success,
    }))
  } else if (msg.cmd === 'close') {
    console.log(JSON.stringify({ ok: true }))
    process.exit(0)
  } else {
    console.log(JSON.stringify({ ok: true }))
  }
})
""",
        encoding="utf-8",
    )
    from pathlib import Path as P
    from fly_policy.mc_env import MinecraftEnv

    root = P(__file__).resolve().parents[2]
    env = MinecraftEnv(
        {
            "minecraft": {"host": "127.0.0.1", "port": 25565, "username": "FruitFly"},
            "mode": "feed",
            "seed_rich": True,
            "census": False,
            "goal": {"spec": str(root / "configs/goals/feed.json")},
        },
        node_script=script,
    )
    hello = env.start()
    assert hello["mode"] == "feed"
    assert hello["seed_rich"] is True
    assert hello["census"] is False
    assert hello["goal_id"] == "feed"
    assert hello["success"]["grn"] == "LB3b"
    env.close()

from dashboard.hub import history, latest, publish, reset


def test_publish_keeps_latest_training_tick():
    reset()
    assert latest() is None
    tick = {
        "t": 1,
        "action": "forward",
        "reward": 0.3,
        "facts": {"rarest": "cherry_log"},
        "frame": {"luminance": [0.1, 0.2], "odor": {"DM1": 0.4}},
        "eye": {"w": 2, "h": 1, "rgb": [120, 85, 48, 46, 110, 42]},
    }
    publish(tick)
    got = latest()
    assert got["action"] == "forward"
    assert got["facts"]["rarest"] == "cherry_log"
    assert got["eye"]["w"] == 2
    assert history()[-1]["reward"] == 0.3

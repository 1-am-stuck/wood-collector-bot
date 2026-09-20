from dashboard.hub import eye, history, latest, publish, publish_stream, reset


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
        "brain": {"n": 3, "h": [0.1, 0.2, 0.9], "names": ["ORN_DL5", "CB_00", "DNp09"]},
    }
    publish(tick)
    got = latest()
    assert got["action"] == "forward"
    assert got["facts"]["rarest"] == "cherry_log"
    assert got["eye"]["w"] == 2
    assert got["brain"]["n"] == 3
    assert history()[-1]["reward"] == 0.3


def test_publish_stream_keeps_packed_first_person_eye():
    reset()
    packed = {"w": 4, "h": 2, "rgb": "AQIDBAUGBwg=", "mode": "first"}
    publish_stream({"stream": "eye", "eye": packed})
    got = eye()
    assert got["w"] == 4
    assert got["h"] == 2
    assert got["mode"] == "first"
    assert got["rgb"] == packed["rgb"]

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from sense.goal_to_sense import apply_goal, goal_success, load_goal


def test_goals_are_injections_not_wood_hardcode():
    empty = {"odor": {}, "taste": {}, "objectChannels": {}, "luminance": [0.4] * 8}
    oak = load_goal(ROOT / "configs/goals/collect_oak.json")
    spruce = load_goal(ROOT / "configs/goals/collect_spruce.json")
    feed = load_goal(ROOT / "configs/goals/feed.json")
    flee = load_goal(ROOT / "configs/goals/flee_creeper.json")

    o = apply_goal(empty, oak, {"position": {"x": 0, "y": 64, "z": 0}, "blocks": [], "entities": []})
    s = apply_goal(empty, spruce, {"position": {"x": 0, "y": 64, "z": 0}, "blocks": [], "entities": []})
    assert o["odor"]["DL5"] > 0
    assert s["odor"]["DC2"] > 0
    assert o["odor"].get("DC2", 0) == 0

    f = apply_goal(empty, feed, None)
    assert f["taste"]["LB3b"] > 0

    quiet = apply_goal(empty, flee, {"position": {"x": 0, "y": 64, "z": 0}, "entities": []})
    assert quiet["objectChannels"].get("LC4", 0) == 0
    loud = apply_goal(empty, flee, {
        "position": {"x": 0, "y": 64, "z": 0},
        "entities": [{"type": "creeper", "name": "creeper", "x": 2, "y": 64, "z": 0}],
    })
    assert loud["objectChannels"]["LC4"] > 0
    assert goal_success(oak, {"inventory": {"oak_log": 1}})
    assert goal_success(feed, {"taste": {"LB3b": 0.5}})
    assert not goal_success(feed, {"inventory": {"cake": 8}})



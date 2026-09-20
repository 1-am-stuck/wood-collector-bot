from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_rarest_yaml_loads_and_catalog_has_nine_logs():
    from train_config import load_goal_catalog, load_train_yaml, spec_for_log
    from sense.rarity import WOOD_LOGS, rarest_log

    cfg = load_train_yaml(ROOT / "configs/train/rarest_minecraft.yaml")
    assert cfg["minecraft"]["required"] is True
    assert cfg["goal"]["selector"] == "rarest_log"
    catalog = load_goal_catalog(cfg["goal"]["catalog"])
    for name in WOOD_LOGS:
        spec = spec_for_log(catalog, name)
        assert spec is not None
        assert spec["success"]["item"] == name
        assert spec["injections"][0]["modality"] == "olfaction"

    rare = rarest_log({"oak_log": 12, "cherry_log": 1, "spruce_log": 4})
    assert rare == "cherry_log"
    assert spec_for_log(catalog, rare)["id"] == "collect:cherry_log"


def test_navigate_yaml_uses_the_full_connectome_and_has_no_goal():
    from train_config import load_train_yaml

    cfg = load_train_yaml(ROOT / "configs/train/navigate_minecraft.yaml")
    assert cfg["graph"]["variant"] == "full"
    assert cfg["mode"] == "navigate"
    assert cfg["run"]["allow_fresh"] is True
    assert not (cfg.get("goal") or {}).get("catalog")
    assert cfg["train"]["minibatch"] == 1
    assert cfg["minecraft"]["required"] is True


def test_feed_yaml_uses_the_full_connectome_and_feed_goal():
    from train_config import load_train_yaml
    from sense.goal_to_sense import load_goal

    cfg = load_train_yaml(ROOT / "configs/train/feed_minecraft.yaml")
    assert cfg["graph"]["variant"] == "full"
    assert cfg["mode"] == "feed"
    assert cfg["run"]["allow_fresh"] is True
    assert cfg["seed_rich"] is True
    assert cfg["census"] is False
    assert cfg["train"]["minibatch"] == 1
    assert cfg["minecraft"]["required"] is True
    out = cfg["run"]["out"]
    assert out.endswith("fly_mc_feed.pt")
    assert not out.endswith("/fly_mc.pt")
    spec = load_goal(cfg["goal"]["spec"])
    assert spec["id"] == "feed"
    assert spec["success"]["type"] == "taste_contact"
    assert spec["success"]["grn"] == "LB3b"

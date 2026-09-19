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

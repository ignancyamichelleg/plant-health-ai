from plant_health.config import _deep_merge, load_config


def test_deep_merge_keeps_unlisted_nested_keys():
    base = {"data": {"image_size": 160, "batch_size": 32}, "seed": 42}
    merged = _deep_merge(base, {"data": {"image_size": 224}})
    assert merged == {"data": {"image_size": 224, "batch_size": 32}, "seed": 42}


def test_load_base_config():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["classes"] == ["angular_leaf_spot", "bean_rust", "healthy"]


def test_experiment_overrides_base():
    cfg = load_config("configs/experiments/baseline_cnn.yaml")
    assert cfg["name"] == "baseline_cnn"
    assert cfg["data"]["image_size"] == 160  # still inherited from base
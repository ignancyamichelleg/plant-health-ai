import pandas as pd

from plant_health.monitoring.reference import build_reference_stats, describe


def test_describe_returns_expected_keys_and_values():
    stats = describe([1, 2, 3, 4, 5])
    assert stats["mean"] == 3.0
    assert stats["p50"] == 3.0
    assert set(stats) == {"mean", "std", "p05", "p50", "p95"}


def test_reference_stats_use_only_train_images():
    manifest = pd.DataFrame(
        {
            "split": ["train", "train", "validation"],
            "label": ["healthy", "bean_rust", "healthy"],
            "readable": [True, True, True],
            "brightness": [100.0, 120.0, 10.0],
            "blur": [500.0, 700.0, 5.0],
        }
    )
    stats = build_reference_stats(manifest)
    assert stats["n_images"] == 2
    assert stats["brightness"]["mean"] == 110.0
    assert stats["class_mix"] == {"healthy": 0.5, "bean_rust": 0.5}
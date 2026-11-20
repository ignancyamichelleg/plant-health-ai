import pandas as pd

from plant_health.data.validate import validate

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
CFG = {
    "classes": CLASSES,
    "validation": {
        "min_side": 64,
        "brightness_min": 30,
        "brightness_max": 230,
        "blur_min": 50,
        "class_share_tolerance": 0.10,
    },
}


def make_df(per_class: int = 2) -> pd.DataFrame:
    rows, n = [], 0
    for split in ("train", "validation", "test"):
        for label in CLASSES:
            for _ in range(per_class):
                n += 1
                rows.append(
                    {
                        "path": f"{split}/{label}/{n}.jpg",
                        "split": split,
                        "label": label,
                        "file_size": 1000,
                        "sha256": f"hash{n}",
                        "readable": True,
                        "width": 500,
                        "height": 500,
                        "channels": 3,
                        "brightness": 120.0,
                        "blur": 800.0,
                    }
                )
    return pd.DataFrame(rows)


def failed_checks(report: dict) -> list[str]:
    return [issue["check"] for issue in report["issues"]]


def test_clean_manifest_passes():
    report = validate(make_df(), CFG)
    assert report["passed"]
    assert report["issues"] == []


def test_unreadable_image_fails():
    df = make_df()
    df.loc[0, ["readable", "width", "height", "channels"]] = [False, None, None, None]
    assert "readable_images" in failed_checks(validate(df, CFG))


def test_unknown_label_fails():
    df = make_df()
    df.loc[0, "label"] = "powdery_mildew"
    assert any("label" in c for c in failed_checks(validate(df, CFG)))


def test_dark_image_fails():
    df = make_df()
    df.loc[0, "brightness"] = 5.0
    assert any("brightness" in c for c in failed_checks(validate(df, CFG)))


def test_blurry_image_fails():
    df = make_df()
    df.loc[0, "blur"] = 1.0
    assert any("blur" in c for c in failed_checks(validate(df, CFG)))


def test_four_channel_image_fails():
    df = make_df()
    df.loc[0, "channels"] = 4
    assert any("channels" in c for c in failed_checks(validate(df, CFG)))


def test_duplicate_across_splits_fails():
    df = make_df()
    df.loc[df["split"] == "test", "sha256"] = df.loc[0, "sha256"]
    assert "no_duplicates_across_splits" in failed_checks(validate(df, CFG))


def test_duplicate_within_split_is_only_a_warning():
    df = make_df()
    df.loc[1, "sha256"] = df.loc[0, "sha256"]  # both in train
    report = validate(df, CFG)
    assert report["passed"]
    assert report["warnings"][0]["check"] == "duplicates_within_split"


def test_missing_class_in_a_split_fails():
    df = make_df()
    df = df[~((df["split"] == "train") & (df["label"] == "healthy"))]
    assert "class_balance" in failed_checks(validate(df, CFG))
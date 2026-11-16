import json

from scripts.make_bdd_subset import build_split


def test_build_split_samples_copies_images_and_normalizes_aliases(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "subset"
    images = source / "images" / "train"
    labels = source / "labels" / "det_20"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    frames = []
    for index in range(4):
        name = f"frame_{index}.jpg"
        (images / name).write_bytes(b"image")
        frames.append({
            "name": name,
            "labels": [{"category": "bike"}, {"category": "motor"}],
        })
    (labels / "det_train.json").write_text(json.dumps(frames), encoding="utf-8")

    assert build_split(source, destination, "train", count=2, seed=42) == 2
    result = json.loads(
        (destination / "labels" / "det_20" / "det_train.json").read_text(encoding="utf-8")
    )
    assert len(result) == 2
    assert all(
        [label["category"] for label in frame["labels"]] == ["bicycle", "motorcycle"]
        for frame in result
    )
    assert all((destination / "images" / "train" / frame["name"]).is_file() for frame in result)


def test_build_split_requires_requested_image_count(tmp_path):
    images = tmp_path / "source" / "images" / "val"
    labels = tmp_path / "source" / "labels" / "det_20"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (labels / "det_val.json").write_text(json.dumps([]), encoding="utf-8")

    try:
        build_split(tmp_path / "source", tmp_path / "subset", "val", count=1)
    except ValueError as exc:
        assert "1 required" in str(exc)
    else:
        raise AssertionError("Expected insufficient images to fail")

import json

from scripts.build_appe_splits import build_appe_splits


def test_clean_splits_do_not_overlap(tmp_path):
    train = [
        {"id": f"train_{i}", "question": f"q{i}", "answer": str(i)}
        for i in range(20)
    ]
    test = [
        {"id": f"test_{i}", "question": f"t{i}", "answer": str(i)}
        for i in range(10)
    ]
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    split_dir = tmp_path / "splits"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    test_path.write_text(json.dumps(test), encoding="utf-8")

    config = {
        "seed": 42,
        "data": {
            "raw_train_path": str(train_path),
            "raw_test_path": str(test_path),
            "split_dir": str(split_dir),
            "calib_size": 3,
            "evo_clean_size": 5,
            "guard_size": 4,
            "test_size": 6,
            "seed": 42,
        },
    }
    splits = build_appe_splits(config)
    train_ids = [
        {row["sample_id"] for row in splits[name]}
        for name in ["calib", "evo_clean", "guard"]
    ]
    assert train_ids[0].isdisjoint(train_ids[1])
    assert train_ids[0].isdisjoint(train_ids[2])
    assert train_ids[1].isdisjoint(train_ids[2])
    assert len(splits["test_clean"]) == 6

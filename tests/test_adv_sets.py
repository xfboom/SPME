from scripts.aaai_utils import build_adv_samples


def test_adv_samples_keep_original_id_and_split_flag():
    clean = [{"sample_id": "clean_1", "question": "1+1?", "answer": "2"}]
    rows = build_adv_samples(
        clean,
        ["numeric_distraction"],
        seed=42,
        is_seen_attack=True,
        prefix="seen",
    )
    assert rows[0]["original_id"] == "clean_1"
    assert rows[0]["answer"] == "2"
    assert rows[0]["attack_type"] == "Numeric_Distraction"
    assert rows[0]["is_seen_attack"] is True


def test_unseen_attack_type_is_not_seen():
    clean = [{"sample_id": "clean_1", "question": "1+1?", "answer": "2"}]
    rows = build_adv_samples(
        clean,
        ["logic_inversion"],
        seed=42,
        is_seen_attack=False,
        prefix="unseen",
    )
    assert rows[0]["attack_type"] == "Logic_Inversion"
    assert rows[0]["is_seen_attack"] is False

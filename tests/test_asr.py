from scripts.aaai_utils import compute_asr


def test_compute_asr_any_wrong():
    clean = [
        {"sample_id": "c1", "original_id": "c1", "is_correct": True},
        {"sample_id": "c2", "original_id": "c2", "is_correct": True},
        {"sample_id": "c3", "original_id": "c3", "is_correct": False},
    ]
    adv = [
        {"sample_id": "a1", "original_id": "c1", "is_correct": True},
        {"sample_id": "a2", "original_id": "c1", "is_correct": False},
        {"sample_id": "a3", "original_id": "c2", "is_correct": True},
        {"sample_id": "a4", "original_id": "c3", "is_correct": False},
    ]
    assert compute_asr(clean, adv) == 0.5


def test_compute_asr_average():
    clean = [
        {"sample_id": "c1", "original_id": "c1", "is_correct": True},
        {"sample_id": "c2", "original_id": "c2", "is_correct": True},
    ]
    adv = [
        {"sample_id": "a1", "original_id": "c1", "is_correct": False},
        {"sample_id": "a2", "original_id": "c1", "is_correct": True},
        {"sample_id": "a3", "original_id": "c2", "is_correct": True},
        {"sample_id": "a4", "original_id": "c2", "is_correct": True},
    ]
    assert compute_asr(clean, adv, mode="average") == 0.25

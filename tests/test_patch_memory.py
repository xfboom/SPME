from memory.AdversarialMemory import AdversarialMemory


def test_patch_memory_save_load(tmp_path):
    memory = AdversarialMemory(base_instruction="Base prompt.")
    memory.add_patch(
        "Numeric_Distraction",
        "Ignore unrelated numbers.",
        attack_type="Numeric_Distraction",
        trigger_condition="irrelevant numbers appear",
    )
    path = tmp_path / "patch_memory.jsonl"
    memory.save_patches_jsonl(str(path))

    loaded = AdversarialMemory(base_instruction="Base prompt.")
    loaded.load_patches_jsonl(str(path))
    assert len(loaded.get_active_patches()) == 1
    assert loaded.get_active_patches()[0].failure_type == "Numeric_Distraction"

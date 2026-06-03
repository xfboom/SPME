import json
import os
import random

def setup_bbh_ablation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_tasks = [
        "boolean_expressions.json",
        "logical_deduction_five_objects.json",
        "causal_judgement.json"
    ]
    
    all_data = []
    
    # 1. 抽取各个子任务的数据
    for task_file in target_tasks:
        file_path = os.path.join(base_dir, "Dataset_format", "BBH", task_file)
        if not os.path.exists(file_path):
            print(f"[Warn] File not found: {file_path}")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        examples = data.get("examples", [])
        
        # 随机抽取最多 200 条
        random.seed(42)
        if len(examples) > 200:
            examples = random.sample(examples, 200)
            
        # 格式化映射为 question / answer
        for item in examples:
            all_data.append({
                "question": item["input"],
                "answer": item["target"],
                "source": task_file
            })
            
    print(f"Total extracted {len(all_data)} samples from {len(target_tasks)} tasks.")
    
    # 2. 打乱并切分数据集 (20% train, 40% val, 40% test)
    random.shuffle(all_data)
    total = len(all_data)
    train_size = int(total * 0.2)
    val_size = int(total * 0.4)
    
    train_set = all_data[:train_size]
    val_set = all_data[train_size:train_size + val_size]
    test_set = all_data[train_size + val_size:]
    
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存结果
    with open(os.path.join(data_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset splits generated in {data_dir}:")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_set)}")

if __name__ == "__main__":
    setup_bbh_ablation()
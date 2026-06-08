import json
from collections import Counter

REQUIRED_FIELDS = {"id", "task_type", "instruction", "input", "reasoning", "output", "tests"}


def main() -> None:
    path = "data/sample_training_data.jsonl"
    count = 0
    task_type_counter: Counter[str] = Counter()
    first_id: str | None = None
    first_instruction: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                raise ValueError(
                    f"Line {count + 1}: missing required fields: {missing}"
                )

            count += 1
            task_type = record["task_type"]
            task_type_counter[task_type] += 1

            if first_id is None:
                first_id = record["id"]
                first_instruction = record["instruction"]

    print(f"Total examples: {count}")
    print("Task type counts:")
    for task_type, cnt in sorted(task_type_counter.items()):
        print(f"  {task_type}: {cnt}")
    if first_id is not None:
        print(f"First example id: {first_id}")
        print(f"First example instruction: {first_instruction}")


if __name__ == "__main__":
    main()

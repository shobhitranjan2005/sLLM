import json
from collections import Counter

REQUIRED_FIELDS = {
    "id",
    "task_type",
    "level",
    "difficulty",
    "skill",
    "instruction",
    "input",
    "reasoning",
    "output",
    "tests",
    "teacher_model",
}


def main() -> None:
    path = "data/hierarchical_reasoning_seed.jsonl"
    count = 0
    difficulty_counter: Counter[str] = Counter()
    level_counter: Counter[int] = Counter()
    skill_counter: Counter[str] = Counter()
    first_id: str | None = None
    first_level: int | None = None
    first_skill: str | None = None
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
            difficulty = record["difficulty"]
            level = record["level"]
            skill = record["skill"]

            difficulty_counter[difficulty] += 1
            level_counter[level] += 1
            skill_counter[skill] += 1

            if first_id is None:
                first_id = record["id"]
                first_level = level
                first_skill = skill
                first_instruction = record["instruction"]

    print(f"Total examples: {count}")
    print("Difficulty counts:")
    for diff, cnt in sorted(difficulty_counter.items()):
        print(f"  {diff}: {cnt}")
    print("Level counts:")
    for lvl, cnt in sorted(level_counter.items()):
        print(f"  {lvl}: {cnt}")
    print("Skill counts:")
    for sk, cnt in sorted(skill_counter.items()):
        print(f"  {sk}: {cnt}")
    if first_id is not None:
        print(f"First example id: {first_id}")
        print(f"First example level: {first_level}")
        print(f"First example skill: {first_skill}")
        print(f"First example instruction: {first_instruction}")


if __name__ == "__main__":
    main()

import json
from collections import Counter

def main() -> None:
    path = "data/sample_training_data.jsonl"
    count = 0
    difficulty_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)  # validate JSON
                count += 1
                difficulty = record.get("difficulty", "unknown")
                difficulty_counter[difficulty] += 1
    print(f"Total examples: {count}")
    print("Examples per difficulty:")
    for diff, cnt in sorted(difficulty_counter.items()):
        print(f"  {diff}: {cnt}")

if __name__ == "__main__":
    main()

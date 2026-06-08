import json

def main() -> None:
    path = "data/sample_training_data.jsonl"
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                json.loads(line)  # validate JSON
                count += 1
    print(f"Total examples: {count}")

if __name__ == "__main__":
    main()

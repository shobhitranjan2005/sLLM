from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset import CodeJsonlDataset, load_jsonl


def main():
    train_path = "data/processed/tiny_train.jsonl"
    val_path = "data/processed/tiny_val.jsonl"

    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    dataset = CodeJsonlDataset(train_path)
    first_text = dataset[0]

    assert "### Problem" in first_text
    assert "### Reasoning" in first_text
    assert "### Algorithm" in first_text
    assert "### Code" in first_text
    assert "### Complexity" in first_text
    assert "### Tests" in first_text

    print("Dataset format is valid.")
    print("First sample preview:")
    print(first_text[:500])


if __name__ == "__main__":
    main()

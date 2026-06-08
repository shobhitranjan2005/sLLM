from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.code_tokenizer import SimpleCodeTokenizer
from src.tokenizer.train_tokenizer import train_tokenizer
from src.utils.dataset import iter_training_texts


TRAIN_PATH = "data/processed/tiny_train.jsonl"
TOKENIZER_PATH = "data/processed/tokenizer/tokenizer.json"


def main() -> None:
    tokenizer = train_tokenizer(TRAIN_PATH, TOKENIZER_PATH, vocab_size=512)

    assert Path(TOKENIZER_PATH).exists(), "Tokenizer file was not created"
    assert len(tokenizer) > 10, "Tokenizer vocab is unexpectedly small"

    loaded = SimpleCodeTokenizer.load(TOKENIZER_PATH)
    sample_text = next(iter_training_texts(TRAIN_PATH))
    ids = loaded.encode(sample_text)
    decoded = loaded.decode(ids)

    assert isinstance(ids, list)
    assert all(isinstance(idx, int) for idx in ids)
    assert ids[0] == loaded.bos_token_id
    assert ids[-1] == loaded.eos_token_id
    assert "Problem" in decoded
    assert "Code" in decoded

    print(f"Tokenizer saved: {TOKENIZER_PATH}")
    print(f"Vocab size: {len(loaded)}")
    print(f"Encoded sample length: {len(ids)}")
    print("Tokenizer check passed.")


if __name__ == "__main__":
    main()

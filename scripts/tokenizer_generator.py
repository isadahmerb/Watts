import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase, Replace, Sequence
from tokenizer_util import export_token_dictionary

SCRIPT_DIR = Path(__file__).resolve().parent

def _build_base_tokenizer(unk_token: str = "<unk>") -> Tokenizer:
    """
    Creates a Rust-backed Hugging Face BPE Tokenizer configured so that
    every pre-token (and therefore every learned token) contains at most
    one space, and only at the very end of the token.
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.normalizer = Sequence([
        Lowercase(),
        Replace(Regex(r"[^\w\s]"), " "),   # pontuação vira espaço
        Replace(Regex(r"\s+"), " "),       # espaços e quebras de linha viram um espaço
    ])
    tokenizer.pre_tokenizer = Split(
        pattern=Regex(r"\w+ ?|[^\w\s] ?|\s"),
        behavior="isolated",
    )
    return tokenizer

def generate_bpe_dictionary(
    text_corpus: Union[str, Path, Iterable[str]],
    vocab_size: int = 4000,
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
    export_path: Optional[Union[str, Path]] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """
    Trains a Rust-accelerated BPE tokenizer using Hugging Face `tokenizers`
    and returns the resulting `token -> id` dictionary.

    If `export_path` or `tokenizer_path` is None, exports directly to the
    same directory as this script (`generator.py`).
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    unk_token = "<unk>" if "<unk>" in special_tokens else special_tokens[0]
    tokenizer = _build_base_tokenizer(unk_token=unk_token)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    if isinstance(text_corpus, Path):
        tokenizer.train([str(text_corpus)], trainer=trainer)
    elif isinstance(text_corpus, str):
        if len(text_corpus) < 512 and Path(text_corpus).is_file():
            tokenizer.train([text_corpus], trainer=trainer)
        else:
            tokenizer.train_from_iterator([text_corpus], trainer=trainer)
    else:
        items = list(text_corpus)
        if items and all(isinstance(x, (str, Path)) and len(str(x)) < 512 and Path(x).is_file() for x in items):
            tokenizer.train([str(x) for x in items], trainer=trainer)
        else:
            tokenizer.train_from_iterator((str(x) for x in items), trainer=trainer)

    dictionary: Dict[str, int] = dict(
        sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    )

    resolved_export_path = (
        SCRIPT_DIR / "bpe_vocab.json" if export_path is None else Path(export_path)
    )
    resolved_tokenizer_path = (
        SCRIPT_DIR / "tokenizer.json" if tokenizer_path is None else Path(tokenizer_path)
    )

    export_token_dictionary(
        dictionary,
        filepath=resolved_export_path,
        tokenizer=tokenizer,
        tokenizer_filepath=resolved_tokenizer_path,
    )

    return dictionary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Rust-backed BPE tokenizer from text files and export the dictionary."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Path(s) to input training text file(s) (e.g. corpus.txt).",
    )
    parser.add_argument(
        "-v", "--vocab-size",
        type=int,
        default=4000,
        help="Target vocabulary size (default: 4000).",
    )
    parser.add_argument(
        "-m", "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency for a BPE merge (default: 2).",
    )
    parser.add_argument(
        "-s", "--special-tokens",
        nargs="+",
        default=["<pad>", "<unk>", "<bos>", "<eos>"],
        help="Special tokens to reserve at the start of the vocabulary.",
    )
    parser.add_argument(
        "-o", "--export-path",
        type=Path,
        default=None,
        help="Output path for bpe_vocab.json (defaults to script directory).",
    )
    parser.add_argument(
        "-t", "--tokenizer-path",
        type=Path,
        default=None,
        help="Output path for tokenizer.json (defaults to script directory).",
    )

    args = parser.parse_args()

    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        parser.error(f"Input file(s) not found: {', '.join(missing)}")

    vocab = generate_bpe_dictionary(
        text_corpus=args.inputs,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        min_frequency=args.min_frequency,
        export_path=args.export_path,
        tokenizer_path=args.tokenizer_path,
    )

    vocab_out = args.export_path or (SCRIPT_DIR / "bpe_vocab.json")
    tok_out = args.tokenizer_path or (SCRIPT_DIR / "tokenizer.json")
    print(f"Done! Trained vocabulary with {len(vocab)} tokens.")
    print(f"Saved dictionary -> {vocab_out}")
    print(f"Saved tokenizer  -> {tok_out}")


if __name__ == "__main__":
    main()
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def export_token_dictionary(
    dictionary: Dict[str, int],
    filepath: Optional[Union[str, Path]] = None,
    tokenizer: Optional[Tokenizer] = None,
    tokenizer_filepath: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Exports the `token -> id` dictionary to JSON, and optionally saves the full
    Hugging Face `Tokenizer` state (`tokenizer.json`) for fast Rust-backed inference.

    If `filepath` or `tokenizer_filepath` is None, defaults to the directory
    containing this script (`tokenizer_util.py`).
    """
    out_path = (
        SCRIPT_DIR / "bpe_vocab.json" if filepath is None else Path(filepath)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_dict = dict(sorted(dictionary.items(), key=lambda kv: kv[1]))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, ensure_ascii=False, indent=2)

    if tokenizer is not None:
        tok_path = (
            SCRIPT_DIR / "tokenizer.json"
            if tokenizer_filepath is None
            else Path(tokenizer_filepath)
        )
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tok_path))

    return out_path


def load_token_dictionary(
    filepath: Optional[Union[str, Path]] = None,
    tokenizer_filepath: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, int], Dict[int, str], Optional[Tokenizer]]:
    """
    Loads the `token -> id` dictionary, builds the reverse `id -> token` map,
    and loads the compiled Rust `Tokenizer` if `tokenizer_filepath` exists.

    If `filepath` or `tokenizer_filepath` is None, defaults to the directory
    containing this script (`tokenizer_util.py`).
    """
    vocab_path = (
        SCRIPT_DIR / "bpe_vocab.json" if filepath is None else Path(filepath)
    )
    with vocab_path.open("r", encoding="utf-8") as f:
        token_to_id: Dict[str, int] = json.load(f)

    id_to_token: Dict[int, str] = {idx: tok for tok, idx in token_to_id.items()}

    tok_path = (
        SCRIPT_DIR / "tokenizer.json"
        if tokenizer_filepath is None
        else Path(tokenizer_filepath)
    )
    tokenizer: Optional[Tokenizer] = None
    if tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))

    return token_to_id, id_to_token, tokenizer


def encode(
    text: str,
    tokenizer: Tokenizer,
    add_special_tokens: bool = False,
) -> List[int]:
    """Encodes text into token IDs using the Rust-backed Hugging Face Tokenizer."""
    return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def encode_batch(
    texts: List[str],
    tokenizer: Tokenizer,
    add_special_tokens: bool = False,
) -> List[List[int]]:
    """Encodes a batch of strings in parallel across CPU threads."""
    return [
        enc.ids
        for enc in tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
    ]


def decode(
    token_ids: List[int],
    tokenizer: Optional[Tokenizer] = None,
    id_to_token: Optional[Dict[int, str]] = None,
    skip_special_tokens: bool = True,
) -> str:
    """
    Decodes token IDs back into text. Uses the Rust `Tokenizer.decode` if provided,
    or falls back to direct dictionary lookup + concatenation.
    """
    if tokenizer is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    if id_to_token is None:
        raise ValueError("Provide either `tokenizer` or `id_to_token` to decode.")

    specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
    pieces = []
    for tid in token_ids:
        tok = id_to_token.get(tid, "")
        if skip_special_tokens and tok in specials:
            continue
        pieces.append(tok)
    return "".join(pieces)
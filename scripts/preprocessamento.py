import re
from tokenizer_util import load_token_dictionary

_token_to_id, _id_to_token, _tokenizer = load_token_dictionary()

if _tokenizer is None:
    raise RuntimeError("tokenizer.json não encontrado em scripts/. Rode o tokenizer_generator.py.")


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)   # pontuação vira espaço
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto + " "                        # toda palavra termina com espaço


def preprocessar(texto: str) -> list[int]:
    """Texto -> lista de IDs de tokens BPE (os 'termos' do TF-IDF)."""
    return _tokenizer.encode(normalizar(texto), add_special_tokens=False).ids


def ver_tokens(texto: str) -> list[str]:
    """Só para depuração: mostra os pedaços de texto."""
    return _tokenizer.encode(normalizar(texto), add_special_tokens=False).tokens
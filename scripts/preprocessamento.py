import re
from tokenizer_util import load_token_dictionary

_token_to_id, _id_to_token, _tokenizer = load_token_dictionary()

if _tokenizer is None:
    raise RuntimeError("tokenizer.json não encontrado em scripts/. Rode o tokenizer_generator.py.")

# Palavras funcionais (artigos, preposições, conjunções, pronomes): quase não carregam
# significado e inflam a similaridade. As palavras interrogativas ficam de fora de propósito.
STOPWORDS = {
    "a", "o", "e", "é", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "ao", "aos", "à", "às",
    "que", "se", "ou", "mas", "já", "me", "te", "meu", "minha", "seu", "sua",
    "eu", "você", "ele", "ela", "isso", "isto", "esse", "essa", "este", "esta",
}
_ids_stop = {i for tok, i in _token_to_id.items() if tok.strip() in STOPWORDS}


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto + " "


def preprocessar(texto: str) -> list[int]:
    """Texto -> IDs de tokens BPE, sem stopwords (os 'termos' do TF-IDF)."""
    ids = _tokenizer.encode(normalizar(texto), add_special_tokens=False).ids
    return [i for i in ids if i not in _ids_stop]


def ver_tokens(texto: str) -> list[str]:
    """Só para depuração: mostra os pedaços de texto."""
    return _tokenizer.encode(normalizar(texto), add_special_tokens=False).tokens
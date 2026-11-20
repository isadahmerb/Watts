import sys
from base import carregar_base, indexar
from preprocessamento import preprocessar
from tfidf import TfIdf
from tokenizer_util import load_token_dictionary

_, id_to_token, _ = load_token_dictionary()
base = indexar(carregar_base())
modelo = TfIdf([e["tokens"] for e in base])

def explicar(pergunta: str, k: int = 2, n: int = 6) -> None:
    tokens = preprocessar(pergunta)
    q = modelo.vetorizar(tokens)
    print("\n", pergunta)
    for i, score in modelo.buscar(tokens, k): 
        contrib = sorted(((q[t] * modelo.vetores[i].get(t, 0.0), t) for t in q), reverse=True)[:n]
        print(f"    {score:.3f} {base[i]['id']} {base[i]['titulo']}")
        print("     ", ", ".join(f"{id_to_token[t].strip()}={c:.3f}" for c, t in contrib if c > 0))

for p in sys.argv[1:]:
    explicar(p)
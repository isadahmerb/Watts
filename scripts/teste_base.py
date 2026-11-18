from collections import Counter
from base import carregar_base, indexar

base = indexar(carregar_base())
print("entradas:", len(base))
print(Counter(e["intencao"] for e in base))
e = base[0]
print(e["id"], e["titulo"], "->", len(e["tokens"]), "tokens")
print(e["perguntas_exemplo"])
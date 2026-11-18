from base import carregar_base, indexar
from preprocessamento import preprocessar
from tfidf import TfIdf

base = indexar(carregar_base())
modelo = TfIdf([e["tokens"] for e in base])

for pergunta in [
    "O que é prontuário das instalações elétricas?",
    "quando a nova norma começa a valer",
    "diferença entre capacitado e habilitado",
    "posso trabalhar energizado sem curso?",
    "como funciona a CIPA?",
]:
    print("\n", pergunta)
    for i, score in modelo.buscar(preprocessar(pergunta)):
        print(f"  {score:.3f}  {base[i]['id']}  {base[i]['titulo']}")
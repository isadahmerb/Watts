import csv
from pathlib import Path
from preprocessamento import preprocessar

CAMINHO_CSV = Path(__file__).resolve().parent.parent / "data" / "base_conhecimento_nr10.csv"


def carregar_base(caminho=CAMINHO_CSV) -> list[dict]:
    entradas = []
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for n, linha in enumerate(csv.DictReader(f, delimiter=";"), start=2):
            if None in linha or None in linha.values():
                raise ValueError(f"Linha {n} ({linha.get('id')}): número de colunas diferente do cabeçalho")
            linha["perguntas_exemplo"] = [
                p.strip() for p in linha["perguntas_exemplo"].split("|") if p.strip()
            ]
            entradas.append(linha)
    return entradas

def texto_indexado(entrada: dict) -> str:
    """O que o chatbot 'enxerga' de cada entrada na hora de comparar com a pergunta."""
    return " ".join([
        entrada["titulo"],
        entrada["palavras_chave"],
        " ".join(entrada["perguntas_exemplo"]),
    ])

def indexar(entradas: list[dict]) -> list[dict]:
    for e in entradas:
        e["tokens"] = preprocessar(texto_indexado(e))
    return entradas
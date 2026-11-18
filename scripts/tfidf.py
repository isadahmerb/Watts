import math
from collections import Counter


class TfIdf:
    def __init__(self, documentos: list[list[int]]):
        """documentos: lista de listas de IDs de tokens (uma por entrada da base)."""
        self.n = len(documentos)

        # df: em quantas entradas cada token aparece
        df = Counter()
        for doc in documentos:
            df.update(set(doc))

        # idf suavizado (evita divisão por zero)
        self.idf = {t: math.log((1 + self.n) / (1 + d)) + 1 for t, d in df.items()}

        self.vetores = [self.vetorizar(doc) for doc in documentos]

    def vetorizar(self, tokens: list[int]) -> dict[int, float]:
        if not tokens:
            return {}
        contagem = Counter(tokens)
        total = len(tokens)
        vetor = {
            t: (c / total) * self.idf[t]
            for t, c in contagem.items()
            if t in self.idf          # token que nunca apareceu na base é ignorado
        }
        norma = math.sqrt(sum(v * v for v in vetor.values()))
        if norma == 0:
            return {}
        return {t: v / norma for t, v in vetor.items()}   # normaliza (comprimento 1)

    @staticmethod
    def cosseno(a: dict[int, float], b: dict[int, float]) -> float:
        # como os vetores já têm comprimento 1, o cosseno é só o produto escalar
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def buscar(self, tokens: list[int], k: int = 3) -> list[tuple[int, float]]:
        q = self.vetorizar(tokens)
        pontuacoes = [(i, self.cosseno(q, v)) for i, v in enumerate(self.vetores)]
        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        return pontuacoes[:k]
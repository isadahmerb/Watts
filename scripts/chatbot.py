import re
from base import carregar_base, indexar
from preprocessamento import preprocessar, normalizar
from tfidf import TfIdf

LIMIAR = 0.25   # score mínimo do melhor resultado (chute inicial)
MARGEM = 0.05   # diferença mínima entre 1º e 2º (chute inicial)

# Regras de segurança: checadas ANTES da similaridade.
REGRAS = [
    ("decisão sobre caso concreto",
     r"\b(posso|pode|devo)\b.*\b(religar|energizar|trabalhar|mexer|abrir|desligar|ligar)\b"),
    ("pedido de parecer de conformidade",
     r"\b(est[aá]|meu|minha)\b.*\b(conforme|seguro|adequad\w*)\b"),
]

AVISO = ("\n\nAtenção: decisões sobre uma instalação ou serviço específico devem ser "
         "tomadas pelo responsável técnico (PLH) ou pelo SESMT da sua organização.")

MSG_FALLBACK = ("Não consegui entender com segurança a sua pergunta, ou ela parece fora do "
                "escopo (NR-10). Tente reformular ou consulte o SESMT / engenheiro de segurança.")

_base = indexar(carregar_base())
_modelo = TfIdf([e["tokens"] for e in _base])

USAR_CONTEXTO = False        # melhoria 2: o avaliar.py liga com --contexto
MAX_PALAVRAS_FOLLOWUP = 5    # pergunta curta = provável continuação

def _eh_followup(pergunta: str) -> bool:
    return len(pergunta.split()) <= MAX_PALAVRAS_FOLLOWUP


def responder(pergunta: str, contexto_anterior: str | None = None) -> dict:
    texto = normalizar(pergunta)
    regras = [nome for nome, padrao in REGRAS if re.search(padrao, texto)]

    consulta = pergunta
    if USAR_CONTEXTO and contexto_anterior and _eh_followup(pergunta):
        consulta = contexto_anterior + " " + pergunta

    (i1, s1), (i2, s2) = _modelo.buscar(preprocessar(consulta), k=2)
    margem = s1 - s2
    entrada = _base[i1]

    if s1 < LIMIAR or margem < MARGEM:
        decisao, resposta = "fallback", MSG_FALLBACK
    elif entrada["intencao"] == "I0_encaminhamento":
        decisao, resposta = "encaminhamento", entrada["resposta"]
    else:
        decisao, resposta = "resposta", entrada["resposta"]
        if regras:
            decisao, resposta = "resposta+aviso", resposta + AVISO

    return {"pergunta": pergunta, "decisao": decisao, "id": entrada["id"],
            "score": round(s1, 3), "margem": round(margem, 3),
            "regras": regras, "resposta": resposta}


if __name__ == "__main__":
    print("Watt (digite 'sair' para encerrar)")
    while True:
        p = input("\nVocê: ").strip()
        if p.lower() == "sair":
            break
        r = responder(p)
        print(f"\nWatt: {r['resposta']}")
        print(f"[{r['decisao']} | {r['id']} | score {r['score']} | margem {r['margem']} | regras {r['regras']}]")
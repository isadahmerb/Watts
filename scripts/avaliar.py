import csv
import re
import sys
from collections import Counter
from pathlib import Path

import chatbot

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_TESTES = RAIZ / "data" / "testes_chatbot_nr10.csv"

INTENCOES = [
    "I0_encaminhamento",
    "I1_documentos",
    "I2_treinamento_habilitacao",
    "I3_nova_nr10_vigencia",
    "I4_visao_geral",
]


def ids_aceitos(trecho_esperado: str) -> set[str]:
    """'I1-01 ou I1-08' -> {I1-01, I1-08}; 'I4-05 (conflito com ...)' -> {I4-05}."""
    return set(re.findall(r"I\d-\d\d", trecho_esperado.split("(")[0]))


def decisoes_aceitas(comportamento: str) -> set[str]:
    if comportamento == "responder":
        return {"resposta", "resposta+aviso"}
    if comportamento == "abster/encaminhar":
        return {"fallback", "encaminhamento"}
    return {"encaminhamento"}   # 'encaminhar' e 'encaminhar (regra de segurança)'


def intencao_obtida(r: dict, base_por_id: dict) -> str:
    """Fallback e encaminhamento contam como I0 (o bot não respondeu o conteúdo)."""
    if r["decisao"] in ("fallback", "encaminhamento"):
        return "I0_encaminhamento"
    return base_por_id[r["id"]]["intencao"]


def main() -> None:
    nomes = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not nomes:
        sys.exit("Uso: python avaliar.py <rotulo> [--contexto] [--limiar=0.25] [--testes=arquivo.csv]")
    rotulo = nomes[0]

    chatbot.USAR_CONTEXTO = "--contexto" in sys.argv
    caminho = CAMINHO_TESTES
    for a in sys.argv[1:]:
        if a.startswith("--limiar="):
            chatbot.LIMIAR = float(a.split("=", 1)[1])
        elif a.startswith("--testes="):
            caminho = RAIZ / "data" / a.split("=", 1)[1]

    base_por_id = {e["id"]: e for e in chatbot._base}

    with open(caminho, encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        campos = leitor.fieldnames
        casos = list(leitor)

    linhas, detalhes = [], []
    for c in casos:
        r = chatbot.responder(c["pergunta"], c["contexto_anterior"] or None)
        obtida = intencao_obtida(r, base_por_id)
        comp = c["comportamento_esperado"]

        comp_ok = r["decisao"] in decisoes_aceitas(comp)
        intencao_ok = obtida == c["intencao_esperada"]
        trecho_ok = (r["id"] in ids_aceitos(c["trecho_esperado"])) if comp == "responder" else None
        ok = comp_ok and intencao_ok

        saida = dict(c)
        saida["intencao_obtida"] = obtida
        saida["trecho_obtido"] = r["id"]
        saida["score_similaridade"] = str(r["score"]).replace(".", ",")
        saida["resposta_obtida"] = r["resposta"]
        saida["resultado"] = "OK" if ok else "ERRO"
        saida["observacao"] = f"decisao={r['decisao']}; margem={r['margem']}; regras={r['regras']}"
        linhas.append(saida)
        detalhes.append((c, r, obtida, comp, comp_ok, intencao_ok, trecho_ok, ok))

    n = len(detalhes)

    # ---- métricas gerais
    acertos = sum(d[7] for d in detalhes)
    resp = [d for d in detalhes if d[3] == "responder"]
    nao_resp = [d for d in detalhes if d[3] != "responder"]
    trechos_ok = sum(1 for d in resp if d[6])
    contexto = "ligado" if chatbot.USAR_CONTEXTO else "desligado"
    print(f"\n=== Avaliação '{rotulo}' | contexto={contexto} | limiar={chatbot.LIMIAR} ===")
    print(f"Taxa de acerto geral (comportamento + intenção): {acertos}/{n} = {acertos/n:.1%}")
    print(f"Trecho exato nas perguntas respondíveis:          {trechos_ok}/{len(resp)} = {trechos_ok/max(len(resp), 1):.1%}")
    print(f"Comportamento correto nas fora de escopo/segurança/encaminhamento: "
          f"{sum(d[4] for d in nao_resp)}/{len(nao_resp)}")
    falsos_fallbacks = [d for d in resp if d[1]["decisao"] in ("fallback", "encaminhamento")]
    respostas_indevidas = [d for d in nao_resp if d[1]["decisao"] in ("resposta", "resposta+aviso")]
    print(f"Abstenções/encaminhamentos indevidos (pergunta respondível): {len(falsos_fallbacks)}")
    print(f"Respostas indevidas (deveria encaminhar/abster): {len(respostas_indevidas)}")
    taxa_enc = sum(d[1]["decisao"] in ("fallback", "encaminhamento") for d in detalhes) / n
    print(f"Taxa de encaminhamento/abstenção do bot: {taxa_enc:.1%}")
    erradas_conf = [d for d in detalhes if d[1]["decisao"] in ("resposta", "resposta+aviso") and not d[7]]
    print(f"Respostas erradas com confiança: {len(erradas_conf)}")

    # ---- por tipo
    print("\nAcerto por tipo de teste:")
    por_tipo = {}
    for d in detalhes:
        t = por_tipo.setdefault(d[0]["tipo"], [0, 0])
        t[0] += d[7]
        t[1] += 1
    for tipo, (a, tot) in por_tipo.items():
        print(f"  {tipo:<16} {a}/{tot}")

    # ---- matriz de confusão
    matriz = {e: Counter() for e in INTENCOES}
    for d in detalhes:
        matriz[d[0]["intencao_esperada"]][d[2]] += 1
    print("\nMatriz de confusão por intenção (linhas = esperado, colunas = obtido):")
    print("        " + "".join(f"{i[:2]:>5}" for i in INTENCOES))
    for e in INTENCOES:
        print(f"  {e[:2]:<6}" + "".join(f"{matriz[e][o]:>5}" for o in INTENCOES))

    print("\nPrecisão e recall por intenção:")
    for i in INTENCOES:
        tp = matriz[i][i]
        fn = sum(matriz[i].values()) - tp
        fp = sum(matriz[e][i] for e in INTENCOES) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {i:<28} precisão {prec:.2f} | recall {rec:.2f}")

    # ---- erros
    print("\nCasos com ERRO:")
    for c, r, obtida, comp, comp_ok, intencao_ok, trecho_ok, ok in detalhes:
        if not ok:
            print(f"  {c['id']} [{c['tipo']}] {c['pergunta'][:60]}")
            print(f"      esperado: {c['trecho_esperado'] or '-'} / {comp}")
            print(f"      obtido:   {r['id']} / {r['decisao']} | score {r['score']} | margem {r['margem']} | 2º {r['segundo']}")
    avisos = [d for d in resp if d[7] and not d[6]]
    if avisos:
        print("\nIntenção certa, mas trecho diferente do esperado:")
        for d in avisos:
            print(f"  {d[0]['id']}: esperado {d[0]['trecho_esperado']} | obtido {d[1]['id']}")

    # ---- salva no formato da planilha
    saida_csv = RAIZ / "data" / f"resultados_{rotulo}.csv"
    with open(saida_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    print(f"\nResultados salvos em {saida_csv}")


if __name__ == "__main__":
    main()
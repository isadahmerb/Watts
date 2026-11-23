import chatbot

# (pergunta anterior, continuação, id esperado). Preencha com 3 ou 4 pares seus.
PARES = [
    ("O que é o PIE?", "E isso é obrigatório?", "I1-03"),
    ("Qual a diferença entre capacitado e habilitado?", "E quem pode autorizar?", "I2-05"),
    # ...
]

for anterior, continuacao, esperado in PARES:
    chatbot.USAR_CONTEXTO = False
    sem = chatbot.responder(continuacao, anterior)
    chatbot.USAR_CONTEXTO = True
    com = chatbot.responder(continuacao, anterior)
    print(f"\n{anterior}  ->  {continuacao}   (esperado {esperado})")
    print(f"  sem contexto: {sem['id']:<6} {sem['decisao']:<14} score {sem['score']}")
    print(f"  com contexto: {com['id']:<6} {com['decisao']:<14} score {com['score']}")
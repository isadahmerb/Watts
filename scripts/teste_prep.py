from preprocessamento import preprocessar, ver_tokens, _tokenizer

print("normalizer:", _tokenizer.normalizer)
print("pre_tokenizer:", _tokenizer.pre_tokenizer)

frase = "Quais são os documentos obrigatórios da NR-10?"
print(ver_tokens(frase))
print(preprocessar(frase))
print(ver_tokens(frase.upper()))   

for frase in [
    "Quais são os documentos obrigatórios da NR-10?",
    "QUAIS SÃO OS DOCUMENTOS OBRIGATÓRIOS DA NR-10?",
    "quais são os documentos obrigatórios da nr-10",
    "Qual o treinamento para trabalhar em instalações elétricas?",
]:
    print(ver_tokens(frase))
# Watt — Chatbot de Documentação do Setor Elétrico

Chatbot em Python que responde dúvidas sobre a **NR-10** (Segurança em Instalações e Serviços em Eletricidade), incluindo a versão vigente e a nova redação (Portaria MTE 737/2026). Projeto integrador A3 de Inteligência Artificial, UNISUL.

> **Aviso:** projeto acadêmico, baseado em empresa e cenários fictícios. Não substitui a consulta ao texto oficial da norma nem a orientação de um profissional de segurança do trabalho.

## Problema

Profissionais do setor elétrico lidam com muita documentação de segurança e costumam ter dúvidas recorrentes: quais documentos são obrigatórios, quem precisa de qual treinamento e o que mudou com a nova NR-10. O Watt dá uma primeira resposta rápida e, quando não tem segurança, **encaminha para um humano** em vez de arriscar uma resposta errada.

## Escopo

| Intenção | Tema |
|---|---|
| I1 | Documentos obrigatórios |
| I2 | Treinamento e habilitação |
| I3 | Nova NR-10 e vigência |
| I4 | Visão geral (conceitos, objetivo e campo de aplicação) |
| Fallback | Pergunta fora do escopo ou de baixa confiança, com encaminhamento |

**Fora do escopo:** outras normas (como a NR-5), documentos internos de empresas e decisões de alto impacto.

## Arquitetura (híbrida)

```
Mensagem → pré-processamento (minúsculas, pontuação, tokenizador BPE)
         → regras de segurança (casos obrigatórios / fora do escopo)
         → TF-IDF + similaridade de cosseno com a base de conhecimento
         → limiar de confiança
              ├─ acima: resposta da base
              └─ abaixo: fallback + encaminhamento humano
```

- **Regras de segurança:** tratam casos que nunca devem ser decididos só por similaridade.
- **TF-IDF + similaridade de cosseno:** recuperação da entrada mais próxima da base (implementação própria, apoiada em scikit-learn).
- **Limiar com fallback:** abaixo do limiar, o bot se abstém.


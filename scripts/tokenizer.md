# Tokenizador BPE Otimizado (Projeto LLM)

Implementação de um tokenizador **Byte Pair Encoding (BPE)** de alta performance escrito em Python utilizando a biblioteca **Hugging Face `tokenizers`** (com núcleo compilado em Rust). Projetado especificamente para treinamento e inferência de modelos de linguagem (LLMs) acadêmicos.

## Características Principais

* **Alta Performance:** Treinamento e codificação paralela multi-thread em Rust através da biblioteca `tokenizers`.

* **Regra de Espaço Único no Final:** Pré-tokenização via expressão regular (`\w+ ?|[^\w\s] ?|\s`) com isolamento de tokens (`behavior="isolated"`), garantindo que nenhum token contenha espaços internos e que todo token possua **no máximo um espaço, exclusivamente no seu último caractere**.

* **Decodificação Sem Perdas (Lossless):** Preserva pontuações, letras maiúsculas/minúsculas e espaços originais, permitindo reconstruir o texto exatamente como era antes da tokenização.

* **Exportação Automática:** Salva tanto o dicionário legível (`bpe_vocab.json`) quanto o estado compilado do tokenizador (`tokenizer.json`) no diretório do script por padrão.

## Requisitos

* **Sistema Operacional:** Linux, macOS ou Windows

* **Python:** Versão `3.8` ou superior

* **Dependências:**

  * `tokenizers` (Hugging Face)

### Instalação Rápida (Linux / macOS)

Crie um ambiente virtual (`venv`), ative-o e instale as dependências com o comando abaixo:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip tokenizers
```

*(No Windows PowerShell, utilize `.venv\Scripts\Activate.ps1` para ativar o ambiente virtual).*

## Estrutura dos Arquivos

* `tokenizer_generator.py`: Script CLI responsável pelo treinamento do modelo BPE e exportação dos arquivos JSON.

* `tokenizer_util.py`: Utilitários para carregar o dicionário, codificar textos (`encode` / `encode_batch`) e decodificar IDs de volta para texto (`decode`).

* `bpe_vocab.json`: Arquivo gerado contendo o mapeamento `token -> id` ordenado.

* `tokenizer.json`: Arquivo gerado contendo a estrutura completa do tokenizador Rust para carregamento rápido.

## Exemplos de Uso

### 1. Treinamento via Linha de Comando (CLI)

O `tokenizer_generator.py` é executado diretamente pelo terminal passando um arquivo, múltiplos arquivos ou pastas inteiras de textos.

**Treinar com um único arquivo (usa tamanho de vocabulário padrão `2048` e frequência mínima `2`):**

```bash
python3 tokenizer_generator.py corpus.txt
```

**Treinar usando todos os arquivos `.txt` de uma pasta `texts/` com ocorrência mínima igual a `3`:**

```bash
python3 tokenizer_generator.py texts/*.txt -m 3
```

*(Caso tenha atualizado o `tokenizer_generator.py` para aceitar diretórios diretamente, basta rodar `python3 tokenizer_generator.py texts -m 3`).*

**Definir tamanho de vocabulário customizado, tokens especiais e caminhos de saída:**

```bash
python3 tokenizer_generator.py texts/*.txt \
  --vocab-size 4096 \
  --min-frequency 3 \
  --special-tokens "<pad>" "<unk>" "<bos>" "<eos>" \
  --export-path saida/bpe_vocab.json \
  --tokenizer-path saida/tokenizer.json
```

**Consultar todas as opções disponíveis na CLI:**

```bash
python3 tokenizer_generator.py --help
```

### 2. Uso no Código Python (Codificação e Decodificação)

Após gerar os arquivos `bpe_vocab.json` e `tokenizer.json` pela CLI, importe as funções de `tokenizer_util.py` no seu pipeline de treinamento ou inferência:

```python
from tokenizer_util import load_token_dictionary, encode, encode_batch, decode

# 1. Carrega o dicionário e o tokenizador Rust do diretório padrão
token_to_id, id_to_token, tokenizer = load_token_dictionary()

# 2. Codificar e decodificar uma única frase
texto_original = "Olá, mundo! Testando o tokenizador da nossa LLM."
ids = encode(texto_original, tokenizer)
texto_reconstruido = decode(ids, tokenizer)

print("IDs:", ids)
print("Tokens:", [id_to_token[i] for i in ids])
print("Reconstruído:", texto_reconstruido)

# 3. Codificação em lote (multi-thread) para pré-processamento de datasets
lote_textos = [
    "Primeira sentença de treinamento.",
    "Segunda sentença com pontuação, números 123 e espaços.",
]
lote_ids = encode_batch(lote_textos, tokenizer)
print("Lote codificado:", lote_ids)
```
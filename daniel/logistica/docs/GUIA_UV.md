# Guia de Transição: Conda para uv

Este documento descreve como migrar o gerenciamento de dependências e ambientes deste projeto do **Conda** para o **uv**, visando maior velocidade e organização.

## 1. O que é o uv?
O `uv` é um gerenciador de pacotes e projetos Python extremamente rápido, escrito em Rust. Ele substitui ferramentas como `pip`, `pip-compile`, `venv`, `poetry` e partes do `conda`.

## 2. Instalação
No Linux ou macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Após a instalação, reinicie o terminal ou execute `source $HOME/.local/bin/env`.

> [!NOTE]
> O executável é instalado em **`/home/<usuario>/.local/bin/uv`**. Se o terminal não reconhecer o comando `uv`, adicione ao seu `~/.bashrc` ou `~/.zshrc`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

## 3. Comandos Rápidos (Comparação)

| Ação | No Conda | No **uv** |
| :--- | :--- | :--- |
| **Criar Ambiente** | `conda create -n logistica` | `uv venv` |
| **Ativar Ambiente** | `conda activate logistica` | `source .venv/bin/activate` |
| **Instalar Pacote** | `conda install <pacote>` | `uv add <pacote>` |
| **Remover Pacote** | `conda remove <pacote>` | `uv remove <pacote>` |
| **Sincronizar Projeto**| `conda env update` | `uv sync` |
| **Listar Pacotes** | `conda list` | `uv pip list` |
| **Desativar Ambiente** | `conda deactivate` | `deactivate` |

## 4. Fluxo de Trabalho no Projeto

Como o projeto já possui um arquivo `pyproject.toml`, o fluxo recomendado é:

1. **Inicializar/Sincronizar:**
   ```bash
   uv sync
   ```
   Isso criará automaticamente uma pasta `.venv` na raiz do projeto e instalará todas as dependências listadas no `pyproject.toml`.

2. **Adicionar novas dependências:**
   ```bash
   uv add seaborn
   ```
   Isso instala o pacote e já o adiciona ao `pyproject.toml`.

3. **Executar Scripts:**
   Você pode rodar scripts sem ativar o ambiente manualmente:
   ```bash
   uv run python seu_script.py
   ```

## 5. Configuração no VS Code / Jupyter
Para usar o ambiente criado pelo `uv` no VS Code:
1. Abra o arquivo `.ipynb`.
2. Clique em **"Select Kernel"** (topo superior direito).
3. Escolha **"Python Environments..."**.
4. Selecione o interpretador localizado em `.venv/bin/python`.

## 6. Por que migrar?
- **Velocidade:** Instalações e resoluções de dependências são quase instantâneas.
- **Ambientes Locais:** Os ambientes ficam dentro da pasta do projeto (`.venv`), facilitando o gerenciamento e a exclusão.
- **Padronização:** Utiliza o padrão moderno `pyproject.toml`.

---

## 7. Troubleshooting

### `uv: comando não encontrado`
O `uv` está instalado mas não está no PATH da sessão atual. Solução:
```bash
export PATH="$HOME/.local/bin:$PATH"
# Para persistir, adicione ao ~/.bashrc ou ~/.zshrc
```
Ou use o caminho completo: `/home/daniel/.local/bin/uv sync`

### `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`
O pandas precisa do `pyarrow` para ler arquivos `.parquet`. Esse pacote não vinha no `pyproject.toml` original.
**Solução:** Já adicionado ao `pyproject.toml`. Basta sincronizar:
```bash
uv sync
```
Isso instalará o `pyarrow` automaticamente no `.venv`.

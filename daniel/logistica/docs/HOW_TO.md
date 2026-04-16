# Guia de Uso do Ambiente e Scripts (AED)

Este guia explica como ativar o ambiente virtual, reexecutar a extração de dados e gerar novos relatórios de análise.

## 1. Ativação do Ambiente

O projeto utiliza o Conda para gerenciar dependências e garantir a compatibilidade com Python 3.13.

```bash
# Ativar o ambiente
conda activate logistica-eda
```

## 2. Reprocessamento de Dados

Se o banco de dados PostgreSQL for atualizado ou se você precisar limpar os dados locais, siga estes passos:

### Extração (DB -> Parquet)
Este script conecta ao PostgreSQL, busca os dados da tabela `pedidos_logistica` e os salva em um arquivo Parquet de alta performance.

```bash
python extract_data.py
```
*Saída esperada: Arquivo `pedidos_logistica.parquet` (~500k linhas).*

### Geração de Profiling (Diagnóstico Automático)
Este script utiliza o `ydata-profiling` para gerar um relatório HTML completo com todas as estatísticas e correlações.

```bash
python eda_profiling.py
```
*Saída esperada: Arquivo `logistica_profiling_report.html`.*

### Limpeza, Normalizações e Engenharia de Features
Este script realiza a limpeza, normalização e engenharia de features do dataset.

```bash
python data_preparation.py
```
*Saída esperada: dataset com as colunas de data corrigidas, variáveis de tempo criadas e registros duplicados removidos. Utilizado no eda_profiling.py, deep_analysis.ipynb e futuro modelo de machine learning*

### Validação da Janela de Risco
Script rápido para validar se as taxas de atraso entre 17h e 18h condizem com o histórico.

```bash
python verify_analysis.py
```

## 3. Uso do Jupyter Notebook

Para análises interativas e visualizações de gráficos:

1. Ative o ambiente: `conda activate logistica-eda`
2. Inicie o Jupyter: `jupyter notebook`
3. Abra o arquivo: `deep_analysis.ipynb`

## 4. Comandos Úteis de Manutenção

### Caso precise reinstalar as dependências:
```bash
pip install -r requirements.txt
```

### Caso precise verificar se o pkg_resources está funcionando:
```bash
python -c "import pkg_resources; print('Ambiente OK')"
```

---
*Nota: Se encontrar o erro `ModuleNotFoundError: No module named 'pkg_resources'`, certifique-se de que o `setuptools` está na versão 69.5.1 (conforme configurado na instalação inicial).*

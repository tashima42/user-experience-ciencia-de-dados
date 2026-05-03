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

### Extração de Cidades (Geolocalização)
Este script extrai a tabela `city_local` com coordenadas geográficas (lat/lon) e nomes normalizados das cidades.

```bash
python extract_city_local.py
```
*Saída esperada: Arquivo `city_local.parquet` (base de municípios).*

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

## 4. Modelagem de Machine Learning (Preditivo)

Este módulo contém modelos configurados para prever atrasos e identificar gargalos na operação logística.

### Random Forest Regressor
Modelo treinado para prever o `dias_atraso_real` (dias de atraso em relação à previsão oficial). Utiliza **Target Encoding** para transportadoras e estados.

```bash
# Opção 1: Com o ambiente já ativado
python train_model-random-forest-regressor.py

# Opção 2: Sem precisar ativar o ambiente manualmente (Conda Run)
conda run -n logistica-eda python train_model-random-forest-regressor.py
```
*Saída esperada: Métricas de erro (MAE, RMSE), gráfico de importância de features em `docs/feature_importance.png` e simulação de risco para pedidos recentes.*

### LightGBM Classification (Risk Prediction)
Modelo otimizado para lidar com desbalanceamento, prevendo a **probabilidade de atraso**. Utiliza o recurso nativo do LightGBM para variáveis categóricas.

```bash
# Opção 1: Com o ambiente ativado
python train_model-lightgbm-classification.py

# Opção 2: Via Conda Run
conda run -n logistica-eda python train_model-lightgbm-classification.py
```
*Saída esperada: Geração de score de risco (Probabilidade), AUC Score e gráfico de importância por Ganho (Gain) em `docs/lightgbm_importance.png`.*

### Integração Preditiva e Dashboard Web (Pipeline Final)
O pipeline final consolida os modelos, analisa interpretabilidade e serve os dados para um Mapa Dinâmico na Web.

**Passo 1: Pré-computar Previsões e Fatores de Risco (SHAP)**
Gera o dataset consolidado em JSON/CSV (incluindo lat/lon e os Top Fatores que causam atraso para cada pedido).
```bash
# Executar para o período desejado (ex: nov-dez/2023) injetando o SHAP
conda run -n logistica-eda python precompute_predictions_v2.py --start 2023-11-15 --end 2023-12-15 --format json --include-shap
```
*(Para mais detalhes e opções, veja o arquivo [PRECOMPUTE_V2.md](PRECOMPUTE_V2.md)).*

**Passo 2: Iniciar o Dashboard (Servidor Web)**
Inicia a aplicação Flask que consome o JSON gerado e exibe o Mapa Logístico interativo.
```bash
# Via UV (mais rápido)
uv run server_map.py

# Ou ativando o ambiente e rodando via Python
conda activate logistica-eda
python server_map.py
```
*Acesse `http://127.0.0.1:8080` no seu navegador.*

## 5. Comandos Úteis de Manutenção

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

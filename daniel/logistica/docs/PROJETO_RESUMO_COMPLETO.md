# Resumo Completo do Projeto: Dashboard Preditiva de Risco Logístico

Este documento consolida toda a jornada de desenvolvimento do projeto de predição de atrasos logísticos, desde a extração inicial dos dados até a implantação do Dashboard interativo com mapas e IA explicável (SHAP).

---

## 1. Extração, Limpeza e Análise Exploratória (EDA)

### A Origem dos Dados
A base de dados original estava hospedada em um servidor PostgreSQL, contendo o histórico de milhares de pedidos (quase 500 mil registros). 
- **Extração:** Utilizamos o script `extract_data.py` para extrair os dados e salvá-los no formato Parquet (`pedidos_logistica.parquet`), o que reduziu drasticamente o tamanho do arquivo e acelerou a leitura.
- **Geolocalização:** Paralelamente, o script `extract_city_local.py` gerou a base de municípios (`city_local.parquet`) contendo Latitude e Longitude para posterior plotagem no mapa.

### Limpeza e Preparação (`data_preparation_final.py`)
A base bruta possuía ruídos. O processo de limpeza padronizou o dataset:
- Conversão rigorosa de strings para minúsculas sem acentuação (para cruzamento de cidades).
- Correção de datas e fusos horários.
- Remoção de valores extremos (outliers) no *Turnaround Time* (`qtd_dias_tat`), preservando o 99º percentil.

### Diagnóstico Estatístico
Geramos relatórios automáticos usando o `ydata-profiling` (`eda_profiling.py`), o que permitiu identificar o altíssimo desbalanceamento da base: **cerca de 97% dos pedidos foram entregues no prazo (Verde) e apenas 3% atrasados (Vermelho/Amarelo)**. Esse insight guiou as escolhas de modelagem subsequentes.

---

## 2. Escolha do Modelo e Engenharia de Features

Foram testadas várias abordagens para lidar com o problema logístico (incluindo Random Forest Regressor para tentar prever *quantos* dias o pedido atrasaria). No entanto, a solução mais assertiva foi tratar como um problema de **Classificação de Risco**.

### O Modelo: LightGBM Classifier (`trabalho3.ipynb`)
A escolha final foi o **LightGBM**. Suas vantagens:
- Lida nativamente com colunas categóricas (como Transportadora, Estado e Tipo de Praça), sem a necessidade de *One-Hot Encoding* gigantescos.
- Extremamente rápido.
- Possui mecanismos embutidos para lidar com datasets desbalanceados.

### Engenharia de Features
Para que o modelo entendesse o comportamento logístico, criamos variáveis derivadas das datas puras:
- **Features Operacionais:** `dias_gastos_cd` (tempo interno), `margem_entrega` (janela real antes do estouro do prazo), `ratio_cd_prazo`.
- **Features Temporais/Sazonais:** `is_fds_despacho` (flag de final de semana), `is_alta_temporada` (Black Friday/Natal), `turno_despacho`.

### Classificação em Semáforo
A saída principal do modelo não é um simples "sim ou não", mas uma **Probabilidade de Atraso**. Transformamos isso em uma regra de negócio visual:
- **🟢 Verde:** Risco < 30% (Confortável)
- **🟡 Amarelo:** Risco de 30% a 70% (Atenção / Acompanhamento)
- **🔴 Vermelho:** Risco > 70% (Ação imediata necessária)

---

## 3. Interpretabilidade com SHAP (Por que vai atrasar?)

Um modelo "caixa preta" não gera confiança. A equipe precisava entender os motivos pelos quais o algoritmo classificava um pedido como "Risco Vermelho". 
Para isso, integramos o **SHAP (SHapley Additive exPlanations)**.

- **Processamento:** O script `shap_updated.py` varre as predições e calcula o impacto individual de cada variável no resultado final daquele pedido.
- **Desafio e Solução das Datas Vazias:** O cálculo inicial do SHAP foi feito descartando pedidos que ainda não tinham "Data de Entrega" (`dt_entrega_pedido`), ignorando cerca de 35 mil pedidos recentes. Por design, nossa inferência prediz o futuro e *não exige* que o pedido esteja entregue. Sendo assim, é esperado que pedidos muito recentes (em trânsito) tenham o alerta de Risco, mas fiquem momentaneamente sem a justificativa detalhada do SHAP (até uma eventual re-geração que não elimine registros em trânsito).

---

## 4. O Motor de Inferência (`precompute_predictions_v2.py`)

Em vez de fazer predições on-the-fly (sob demanda) que travariam o servidor web, desenvolvemos um pipeline de pré-computação em lote (batch).

### O Fluxo:
1. **Filtro Temporal:** Lê apenas o dataset de inferência (Holdout 30% que o modelo nunca viu no treino).
2. **Predição em Lote:** Aplica o modelo LightGBM para o período solicitado via parâmetros `--start` e `--end`.
3. **Enriquecimento:** Cruza os dados com `city_local.parquet` para injetar a Latitude e Longitude do destino.
4. **Integração do SHAP:** Caso executado com a flag `--include-shap`, o script automaticamente faz um _merge_ em memória dos Top 4 Fatores ofensores de cada pedido.
5. **Exportação:** Salva tudo em um único arquivo `precomputed_predictions_v2.json`, perfeitamente otimizado para a Web.

---

## 5. Visualização: Dashboard Interativo (Mapas e Web)

Por fim, os dados pré-processados precisavam ser consumidos de forma gerencial e ágil. 
Desenvolvemos duas frentes de Frontend: um Dashboard Analítico (Dash) e um Mapa Dinâmico (Flask).

### 5.1 O Dashboard Analítico (`dash.py`)
Criado na branch 'main', esta interface foi construída em **Dash/Plotly**. 
- O frontend consome os dados através da API no `server.py`.
- Renderiza tabelas de pedidos em risco e gráficos analíticos.
- Assim como o mapa, se beneficia dos modelos treinados em `trabalho3.ipynb` e dos cálculos de interpretabilidade de `shap_updated.py` para exibir gráficos detalhados dos fatores ofensores de cada estado/transportadora.

### 5.2 O Mapa Dinâmico (Flask + MapLibre)
A aplicação voltada para a distribuição geográfica foi construída usando **Flask (`server_map.py`)** e renderiza templates HTML customizados (`risk_mapa.html` e `risk_mapa_cluster.html`).
- **MapLibre GL:** Usado para renderizar o mapa interativo do Brasil.
- **Visualização Flexível:** O mapa possui opções (Toggle) para mapa rodoviário e "Light Gray Canvas" (visão mais limpa e executiva).
- **Z-Index Inteligente e Ondas (Ripple):** Pedidos com risco crítico (Vermelho) sempre ficam sobrepostos aos verdes, e possuem uma animação de "pulsar/ondas" para atrair a atenção do gestor de logística.
- **Visão Clusterizada:** Na rota `/cluster`, o sistema agrupa municípios próximos de acordo com o zoom, revelando o status predominante de Risco da região (Agrupamento Dinâmico).

### O Painel Lateral Dinâmico
Ao clicar em uma cidade no mapa (que pode ter dezenas de pedidos), o sistema localiza imediatamente o **Maior Risco de Atraso para esse Destino**.
O painel exibe:
1. O Status geral da cidade (distribuição Verde/Amarelo/Vermelho).
2. O código do "Pior" pedido e por qual Transportadora/CD ele foi emitido.
3. **O Gráfico SHAP HTML:** Se os dados do SHAP estiverem disponíveis para o pedido, são renderizadas mini-barras horizontais proporcionais na tela. Barras em laranja (#ff7f0e) demonstram graficamente quais variáveis estão puxando a probabilidade de atraso para cima (ex: `margem_entrega` ou `is_alta_temporada`).

---

### Resumo Executivo
Toda a complexidade de Machine Learning, Engenharia de Dados, Pandas e SHAP foi encapsulada nos scripts de background. A área de negócio interage apenas com um Mapa visual rápido, fluído e que entrega "mastigado": **Aonde focar a atenção hoje, em qual pedido, e por qual motivo.**

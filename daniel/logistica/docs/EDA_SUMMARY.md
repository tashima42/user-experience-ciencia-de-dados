# Relatório de Início da Fase AED (Análise Exploratória de Dados)

## Contexto do Projeto
Este documento registra a transição da fase de estruturação de dados (PostgreSQL) para a fase de Análise Exploratória profunda em Python, visando a futura criação de um modelo preditivo de riscos de atraso na logística.

## 1. Configuração do Ambiente Técnico
- **Stack**: Python 3.13 (Conda Environment: `logistica-eda`).
- **Principais Bibliotecas**: `pandas`, `ydata-profiling`, `sqlalchemy`, `matplotlib`, `seaborn`, `pyarrow`.
- **Extração**: Dados extraídos da tabela `pedidos_logistica` (~500k registros) e convertidos para o formato **Parquet** (`pedidos_logistica.parquet`) para otimização de performance.

## 2. Validação da "Janela de Risco" (17h - 18h)
O objetivo era validar o insight prévio obtido via SQL de que pedidos despachados neste horário possuem maior taxa de atraso.

### Definições de Atraso e Discrepâncias Identificadas:
Durante a análise, identificamos duas formas de medir o atraso, resultando em métricas significativamente diferentes:

1. **Visão Estrita (dt_entrega > dt_previsao)**:
   - Taxa de atraso na janela das 17h: **~27%**.
   - Esta métrica ignora regras de tolerância ou feriados, sendo muito mais rigorosa.

2. **Visão Operacional (performance_entrega == 'Fora do Prazo')**:
   - Taxa de atraso na janela das 17h: **~7.05%**.
   - Esta métrica valida o insight inicial (~7.6%) e sugere que a operação possui critérios de "sucesso" que aceitam certas variações de data.

**Conclusão**: O aumento de atraso às 17h é real e estatisticamente significante em ambas as métricas, sendo um ponto focal crítico para o time operacional.

## 3. Artefatos de Análise
Os seguintes arquivos foram gerados para suporte à decisão:
- `deep_analysis.ipynb`: Notebook com visualizações da taxa de atraso por hora, transportadora e UF.
- `logistica_profiling_report.html`: Relatório interativo completo (`ydata-profiling`) com distribuição de todas as variáveis.

## 4. Próximos Passos (Estratégia UX/DS)
1. **Feature Engineering**: Criar variáveis que capturem a volumetria horária por transportadora.
2. **Normalização Geográfica**: Refinar a análise de cidades para identificar polos de baixa performance.
3. **Seleção de Modelos**: Iniciar testes com algoritmos de classificação para prever o risco no momento do despacho.

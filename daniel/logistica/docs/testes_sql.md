```sql

--- verificando por janelas de registros
select distinct (lower(cidade_destinatario_normalizada)) from pedidos_logistica
order by 1
limit 1000 offset 5000; -- depois pular offset de 1000..


select count(*) from staging_pedidos
where dt_despacho_pedido is null -- e hr_despacho_pedido
-- 6756


select count(*) from staging_pedidos
where qtd_dias_tat is null -- tp_performance_entrega = 'Entregue no Prazo'
-- 35998

select distinct(tp_performance_entrega) from staging_pedidos
--Entregue no Prazo
--Fora do Prazo


select distinct(uf) from staging_pedidos
-- 27 como esperado


select distinct(grp_transportadora) from staging_pedidos
-- Transportadora 1 a 8


select distinct(tp_praca) from pedidos_logistica;      
-- Capital
-- Interior
-- Reg. Metropolitana


select distinct(unidade_negocio) from pedidos_logistica;
-- Mono
-- Multi


select distinct(cd_origem) from pedidos_logistica;     
-- PR-Campina G. Sul
-- SP-Cajamar
-- SP-Jaguaré
-- SP-Registro


-- Com problemas:
select count(distinct (lower(cidade_destinatario))) from staging_pedidos
-- 6542  (tem lixos e replicados)


-- Normalizado e alguns corrigidos
select count(distinct (lower(cidade_destinatario_normalizada))) from pedidos_logistica
-- 5703 (quase o esperado)


-- procurar duplicados
WITH CTE AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY cidade_destinatario_normalizada, uf, grp_transportadora, dt_despacho, dt_entrega,
            dt_previsao, dt_criacao, dt_pagamento, flg_ocorrencias, tp_praca, unidade_negocio,
            cd_origem, qtd_dias_tat, performance_entrega -- Colunas que definem a duplicidade
            ORDER BY cod_pedido -- Ordenar para decidir qual manter (ex: menor ID)
        ) AS row_num
    FROM 
        pedidos_logistica
)
SELECT * FROM CTE WHERE row_num > 1;



-- Identificando a "janela de risco" de despacho
SELECT 
    EXTRACT(HOUR FROM dt_despacho) as hora_dia,
    COUNT(*) as total_pedidos,
    ROUND(AVG(CASE WHEN performance_entrega != 'Entregue no Prazo' THEN 1 ELSE 0 END) * 100, 2) as taxa_atraso
FROM pedidos_logistica
GROUP BY 1 ORDER BY 1;
-- Esse tipo de insight (ex: pedidos despachados após as 17h têm 7% mais chance de atraso) 
-- é ouro para o time de logística e você consegue em segundos após o dump.

```

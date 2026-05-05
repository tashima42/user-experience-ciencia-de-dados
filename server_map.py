from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions_v2.json")

app = Flask(__name__)

MAP_DATA_CACHE = None

def get_dynamic_period():
    json_path = os.path.join(BASE_DIR, "data", "precomputed_predictions_v2.json")
    try:
        if os.path.exists(json_path):
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dates = [row.get("dt_criacao", "")[:10] for row in data if row.get("dt_criacao")]
            if dates:
                min_dt = min(dates)
                max_dt = max(dates)
                # Convert 'YYYY-MM-DD' to 'DD/MM/YYYY'
                def fmt(d):
                    parts = d.split('-')
                    if len(parts) == 3:
                        return f"{parts[2]}/{parts[1]}/{parts[0]}"
                    return d
                return f"{fmt(min_dt)} - {fmt(max_dt)}"
    except Exception as e:
        print(f"Error reading dates: {e}")
    return "Período Indisponível"

@app.route("/")
def mapa():
    """Map dashboard page."""
    return render_template("risk_mapa.html", period=get_dynamic_period())


@app.route("/cluster")
def mapa_cluster():
    """Clustered map dashboard page."""
    return render_template("risk_mapa_cluster.html", period=get_dynamic_period())



@app.route("/api/map_data", methods=["GET"])
def api_map_data():
    """Get aggregated map data by city from JSON."""
    global MAP_DATA_CACHE
    if MAP_DATA_CACHE is not None:
        return jsonify(MAP_DATA_CACHE)
    
    import json
    json_path = os.path.join(BASE_DIR, "data", "precomputed_predictions_v2.json")
    if not os.path.exists(json_path):
        return jsonify([])
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cities = {}
        for row in data:
            if row.get("lat") is None or row.get("lon") is None:
                continue
                
            city = row.get("cidade_destinatario", "")
            uf = row.get("uf", "")
            key = f"{city}_{uf}"
            
            risk = row.get("risco_semaforo", "")
            prob = row.get("probabilidade_atraso", 0)
            
            if key not in cities:
                cities[key] = {
                    "cidade_destinatario": city,
                    "uf": uf,
                    "lat": row.get("lat"),
                    "lon": row.get("lon"),
                    "total_pedidos": 0,
                    "qtd_verde": 0,
                    "qtd_amarelo": 0,
                    "qtd_vermelho": 0,
                    "pior_risco_nome": risk,
                    "pior_cor": row.get("cor_semaforo", "#ccc"),
                    "max_probabilidade": prob,
                    "pior_pedido_id": row.get("cod_pedido") or row.get("id"),
                    "pior_transportadora": row.get("grp_transportadora", ""),
                    "pior_cd_origem": row.get("des_cd_origem", ""),
                    "pior_dt_despacho": row.get("dt_despacho_pedido", ""),
                    "pior_dt_previsao": row.get("dt_previsao_entrega_cliente", ""),
                    "pior_shap_columns": row.get("top_shap_columns", ""),
                    "pior_shap_values": row.get("top_shap_values", "")
                }
                
            c = cities[key]
            c["total_pedidos"] += 1
            
            if "Verde" in risk:
                c["qtd_verde"] += 1
            elif "Amarelo" in risk:
                c["qtd_amarelo"] += 1
            elif "Vermelho" in risk:
                c["qtd_vermelho"] += 1
                
            # Update worst order info if this one has higher probability of delay
            if prob > c["max_probabilidade"]:
                c["max_probabilidade"] = prob
                c["pior_risco_nome"] = risk
                c["pior_cor"] = row.get("cor_semaforo", c["pior_cor"])
                c["pior_pedido_id"] = row.get("cod_pedido") or row.get("id")
                c["pior_transportadora"] = row.get("grp_transportadora", "")
                c["pior_cd_origem"] = row.get("des_cd_origem", "")
                c["pior_dt_despacho"] = row.get("dt_despacho_pedido", "")
                c["pior_dt_previsao"] = row.get("dt_previsao_entrega_cliente", "")
                c["pior_shap_columns"] = row.get("top_shap_columns", "")
                c["pior_shap_values"] = row.get("top_shap_values", "")
                
            # Determine overall city color (Worst color present)
            if c["qtd_vermelho"] > 0:
                c["cor_cidade"] = "#E74C3C"  # Vermelho
                c["status_cidade"] = "🔴 Vermelho"
            elif c["qtd_amarelo"] > 0:
                c["cor_cidade"] = "#F1C40F"  # Amarelo
                c["status_cidade"] = "🟡 Amarelo"
            else:
                c["cor_cidade"] = "#2ECC71"  # Verde
                c["status_cidade"] = "🟢 Verde"
                
        aggregated = list(cities.values())
        MAP_DATA_CACHE = aggregated
        return jsonify(aggregated)
    except Exception as e:
        print(f"Error loading map data: {e}")
        return jsonify([])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)

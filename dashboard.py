"""Dashboard de mantenimiento predictivo - AI4I 2020
Ejecutar:  python -m poetry run streamlit run dashboard.py
Antes hay que correr Proyecto.ipynb (crea la carpeta dashboard_datos/).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATOS = Path(__file__).parent / "dashboard_datos"
RANDOM_STATE = 42
VARS = ["temp_aire", "temp_proceso", "velocidad", "torque", "desgaste", "delta_temp", "potencia", "esfuerzo"]
TEAL, ROJO, GRIS = "#00737E", "#B03A2E", "#9AA5AB"

st.set_page_config(page_title="Mantenimiento predictivo CNC", page_icon="🛠️", layout="wide")


# ---------------- Carga de datos y modelo ----------------
@st.cache_data
def cargar():
    df = pd.read_csv(DATOS / "datos.csv")
    metricas = pd.read_csv(DATOS / "metricas.csv", index_col=0)
    perfil = pd.read_csv(DATOS / "perfil_clusters.csv", index_col=0)
    pca = pd.read_csv(DATOS / "comparacion_pca.csv", index_col=0)
    lda = pd.read_csv(DATOS / "comparacion_lda.csv", index_col=0)
    params = json.load(open(DATOS / "modelo_final.json"))
    return df, metricas, perfil, pca, lda, params


@st.cache_resource
def entrenar(df, params):
    # Mismo split y parámetros que el notebook (AdaBoost GridSearch)
    X = pd.get_dummies(df[VARS + ["tipo"]], columns=["tipo"], dtype=int)
    y = df["falla"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    modelo = Pipeline([("sc", StandardScaler()),
                       ("m", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=params["max_depth"]),
                                                n_estimators=params["n_estimators"],
                                                learning_rate=params["learning_rate"], random_state=RANDOM_STATE))])
    modelo.fit(X_train, y_train)
    return modelo, X.columns, X_test, y_test


df, metricas, perfil, comp_pca, comp_lda, params = cargar()
modelo, columnas, X_test, y_test = entrenar(df, params)

# ---------------- Filtros ----------------
st.sidebar.header("Filtros")
tipos = st.sidebar.multiselect("Tipo de producto", ["L", "M", "H"], default=["L", "M", "H"])
rango = st.sidebar.slider("Desgaste de la herramienta (min)", 0, int(df["desgaste"].max()), (0, int(df["desgaste"].max())))
d = df[df["tipo"].isin(tipos) & df["desgaste"].between(*rango)]
st.sidebar.caption(f"{len(d):,} de {len(df):,} ciclos seleccionados".replace(",", "."))
st.sidebar.markdown("---")
st.sidebar.caption("Datos: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0). Datos sintéticos.")

st.title("🛠️ Mantenimiento predictivo de una máquina CNC")
st.caption("Data Analytics en Gestión Industrial · Maestría en Gestión Industrial · Universidad de Ibagué")

if d.empty:
    st.warning("No hay datos con esos filtros.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Estado actual", "🔍 Hallazgos", "🤖 Modelos", "🎛️ Simulador", "✅ Plan de acción"])

# ---------------- 1. Estado actual ----------------
with tab1:
    fallas = int(d["falla"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fallas", fallas)
    k2.metric("Tasa de falla", f"{d['falla'].mean() * 100:.2f} %")
    k3.metric("MTBF (ciclos entre fallas)", f"{len(d) / fallas:.1f}" if fallas else "—")
    k4.metric("Fallas con desgaste > 200 min", int(d.loc[d["desgaste"] > 200, "falla"].sum()))

    c1, c2 = st.columns(2)
    modos = d.loc[d["falla"] == 1, "tipo_falla"].value_counts().sort_values()
    c1.plotly_chart(px.bar(modos, orientation="h", title="Fallas por modo", color_discrete_sequence=[ROJO])
                    .update_layout(showlegend=False, xaxis_title="Fallas", yaxis_title=""), width="stretch")
    por_tipo = (d.groupby("tipo")["falla"].mean() * 100).reindex(["L", "M", "H"]).dropna().reset_index()
    c2.plotly_chart(px.bar(por_tipo, x="tipo", y="falla", title="Tasa de falla por tipo de producto (%)",
                           text_auto=".1f", color_discrete_sequence=[TEAL])
                    .update_layout(xaxis_title="Tipo", yaxis_title="Tasa de falla (%)"), width="stretch")
    st.info("Los productos L (calidad baja) fallan casi el doble que los H. Solo el 3,3 % de los ciclos son fallas.")

# ---------------- 2. Hallazgos ----------------
with tab2:
    c1, c2 = st.columns(2)
    tasa_desg = (d.groupby(["rango_desgaste", "tipo"], observed=True)["falla"].mean() * 100).reset_index()
    orden = ["0-50", "51-100", "101-150", "151-200", ">200"]
    c1.plotly_chart(px.bar(tasa_desg, x="rango_desgaste", y="falla", color="tipo", barmode="group",
                           category_orders={"rango_desgaste": orden, "tipo": ["L", "M", "H"]},
                           title="Tasa de falla por desgaste (%)")
                    .update_layout(xaxis_title="Desgaste (min)", yaxis_title="Tasa de falla (%)"), width="stretch")
    c1.caption("Pasando 200 min de desgaste la tasa de falla se dispara: cambiar la herramienta antes.")

    muestra = pd.concat([d[d["falla"] == 0].sample(min(2000, (d["falla"] == 0).sum()), random_state=1), d[d["falla"] == 1]])
    fig = px.scatter(muestra, x="velocidad", y="torque", color="tipo_falla", opacity=0.7,
                     title="Zona de operación y modo de falla",
                     color_discrete_map={"Sin falla": "lightgray"})
    c2.plotly_chart(fig.update_traces(marker_size=5), width="stretch")
    c2.caption("Cada modo de falla ocupa una zona distinta; el centro de la curva es la zona segura.")

    st.subheader("Regímenes de operación (K-Means, k = 4)")
    p = perfil.rename(columns={"falla": "tasa_falla_%"}).round(1).sort_values("tasa_falla_%", ascending=False)
    nombres = {p.index[0]: "Herramienta gastada", p.index[1]: "Poca disipación de calor",
               p.index[2]: "Alta velocidad / bajo torque", p.index[3]: "Operación normal"}
    p.insert(0, "régimen", [nombres[i] for i in p.index])
    c1, c2 = st.columns([3, 2])
    c1.dataframe(p, width="stretch")
    c2.plotly_chart(px.bar(p, x="régimen", y="tasa_falla_%", color_discrete_sequence=[TEAL], text_auto=".1f",
                           title="Tasa de falla por régimen (%)").update_layout(xaxis_title="", yaxis_title=""),
                    width="stretch")

# ---------------- 3. Modelos ----------------
with tab3:
    st.subheader("Comparación de modelos (conjunto de prueba)")
    enteros = ["Fallas no detectadas", "Falsas alarmas"]
    fmt = {c: ("{:.0f}" if c in enteros else "{:.3f}") for c in metricas.columns}
    st.dataframe(metricas.style.format(fmt).highlight_max(subset=["F1", "Recall"], color="#E3F1F2")
                 .highlight_min(subset=["Fallas no detectadas"], color="#E3F1F2"), width="stretch")
    c1, c2 = st.columns(2)
    top = metricas.sort_values("F1").reset_index(names="modelo")
    c1.plotly_chart(px.bar(top, x="F1", y="modelo", orientation="h", title="F1 por modelo",
                           color_discrete_sequence=[TEAL]).update_layout(yaxis_title=""), width="stretch")

    pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    c2.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              x=["Sin falla", "Falla"], y=["Sin falla", "Falla"],
                              labels=dict(x="Predicho", y="Real"), title=f"Matriz de confusión – {params['modelo']} (modelo final)"),
                    width="stretch")
    tn, fp, fn, tp = cm.ravel()
    c2.metric("Paradas no programadas en prueba", f"{fn} con el modelo", f"-{tp} frente a {fn + tp} sin modelo",
              delta_color="inverse")
    c2.caption(f"Costo del error: cada falla no detectada es una parada no programada; cada falsa alarma ({fp}) "
               "es solo una inspección preventiva.")

    imp = pd.Series(modelo.named_steps["m"].feature_importances_, index=columnas).sort_values().reset_index()
    imp.columns = ["variable", "importancia"]
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.bar(imp, x="importancia", y="variable", orientation="h", title="Importancia de variables",
                           color_discrete_sequence=[TEAL]).update_layout(yaxis_title=""), width="stretch")
    c2.plotly_chart(px.bar(comp_pca.rename_axis("Modelo").reset_index().melt(id_vars="Modelo"), x="Modelo", y="value", color="variable",
                           barmode="group", title="F1 con y sin PCA").update_layout(yaxis_title="F1", legend_title=""),
                    width="stretch")
    c3.plotly_chart(px.bar(comp_lda.rename_axis("Modelo").reset_index().melt(id_vars="Modelo"), x="Modelo", y="value", color="variable",
                           barmode="group", title="F1 con y sin LDA").update_layout(yaxis_title="F1", legend_title=""),
                    width="stretch")
    st.info("Los modelos de árboles (AdaBoost, Random Forest) son los mejores porque la falla depende de umbrales "
            "(desgaste > 200 min, esfuerzo > 11.000). PCA y LDA ayudan a interpretar, pero bajan el F1 como preprocesamiento.")

# ---------------- 4. Simulador ----------------
with tab4:
    st.subheader("¿Va a fallar la máquina en este ciclo?")
    st.caption("Ajusta las condiciones de operación y el modelo final (AdaBoost) calcula un puntaje de riesgo. "
               "Con 50 % o más el modelo predice falla. AdaBoost concentra sus puntajes cerca del 50 %: "
               "un ciclo normal ronda el 20-35 % y una falla el 60-75 %.")
    c1, c2, c3 = st.columns(3)
    tipo = c1.selectbox("Tipo de producto", ["L", "M", "H"])
    temp_aire = c1.slider("Temperatura del aire (K)", 295.0, 305.0, 300.0, 0.1)
    temp_proceso = c1.slider("Temperatura del proceso (K)", 305.0, 314.0, 310.0, 0.1)
    velocidad = c2.slider("Velocidad (rpm)", 1150, 2900, 1500, 10)
    torque = c2.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    desgaste = c3.slider("Desgaste actual (min)", 0, 255, 100)

    fila = {"temp_aire": temp_aire, "temp_proceso": temp_proceso, "velocidad": velocidad, "torque": torque,
            "desgaste": desgaste, "delta_temp": temp_proceso - temp_aire,
            "potencia": torque * velocidad * 2 * np.pi / 60, "esfuerzo": torque * desgaste,
            "tipo_H": int(tipo == "H"), "tipo_L": int(tipo == "L"), "tipo_M": int(tipo == "M")}
    x = pd.DataFrame([fila])[columnas]
    prob = modelo.predict_proba(x)[0, 1]

    c3.metric("Puntaje de riesgo", f"{prob * 100:.1f} %")
    if prob >= 0.5:
        c3.error("⚠️ El modelo predice FALLA: programar mantenimiento preventivo")
    elif prob >= 0.38:
        c3.warning("Riesgo medio: vigilar el ciclo")
    else:
        c3.success("Riesgo bajo: operación normal")

    alertas = []
    if desgaste > 200: alertas.append("Desgaste mayor a 200 min: cambiar la herramienta.")
    if fila["esfuerzo"] > 11000: alertas.append("Esfuerzo mayor a 11.000: reducir el torque.")
    if fila["delta_temp"] < 8.6: alertas.append("delta_temp menor a 8,6 K: revisar la refrigeración.")
    if not 3500 <= fila["potencia"] <= 9000: alertas.append("Potencia fuera de 3.500-9.000 W: riesgo de falla de potencia.")
    st.markdown("**Variables calculadas:** " f"delta_temp = {fila['delta_temp']:.1f} K · potencia = {fila['potencia']:,.0f} W · "
                f"esfuerzo = {fila['esfuerzo']:,.0f}".replace(",", "."))
    for a in alertas:
        st.markdown(f"- {a}")

# ---------------- 5. Plan de acción ----------------
with tab5:
    st.subheader("Plan de acción")
    st.dataframe(pd.DataFrame({
        "Acción": ["Cambiar la herramienta a los 190 min", "Mantenimiento preventivo guiado por el modelo",
                   "Limitar el torque en productos L", "Revisar la refrigeración si delta_temp < 8,6 K",
                   "Operar entre 3.500 y 9.000 W"],
        "Respaldo": ["La falla se dispara después de 200 min", "Detecta cerca de 9 de cada 10 fallas",
                     "Con esfuerzo > 11.000 la falla llega al 82 %", "En esa condición la falla llega al 17 %",
                     "Las fallas de potencia ocurren fuera de ese rango"],
        "Costo": ["Bajo", "Medio", "Bajo", "Medio", "Bajo"],
        "Riesgo": ["Más consumo de herramientas", "Algunas inspecciones innecesarias", "Menor ritmo de producción en L",
                   "Paradas cortas de revisión", "Ajustar programas de corte"],
        "Seguimiento": ["Fallas por desgaste y sobreesfuerzo", "Recall y paradas no programadas por mes",
                        "Fallas por sobreesfuerzo", "Fallas por disipación de calor", "Fallas por potencia"],
    }), hide_index=True, width="stretch")
    st.success("**Meta:** bajar la tasa de falla del 3,3 % a menos del 1,5 % en 6 meses.")
    st.caption("Limitaciones: dataset sintético de una sola máquina; validar con datos reales de la planta antes de implementar.")

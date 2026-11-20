from pathlib import Path

import pandas as pd
import streamlit as st


archivo = Path(__file__).resolve().parents[2] / "outputs" / "metricas_prototipo.csv"
metricas = pd.read_csv(archivo)

st.title("Métricas del prototipo")
st.caption("Resultados de las pruebas documentadas en Knowledge_Hub_Metricas.ipynb.")

componente = st.selectbox(
    "Componente",
    ["Todos", *metricas["Componente"].drop_duplicates()],
)

if componente == "Todos":
    seleccion = metricas
else:
    seleccion = metricas.loc[metricas["Componente"] == componente]

st.dataframe(seleccion, hide_index=True, width="stretch")

st.subheader("Pruebas funcionales del chatbot")
st.markdown(
    "- **Dato disponible.** Indicó que las vacaciones deben solicitarse "
    "con al menos dos semanas de anticipación y citó la política `PP08`.\n"
    "- **Dato ausente.** Respondió «Información insuficiente» al preguntar "
    "cuántos días corresponden por cuatro años de antigüedad."
)

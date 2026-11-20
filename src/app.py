import streamlit as st

from recommender import DocumentRecommender


st.set_page_config(page_title="Softtek Knowledge Hub", layout="wide")


@st.cache_resource
def cargar_recomendador():
    return DocumentRecommender()


st.title("Softtek Knowledge Hub")
st.header("Documentos relacionados")

motor = cargar_recomendador()
documentos = motor.docs.set_index("id")
ids = documentos.index.tolist()

seleccion = st.selectbox(
    "Documento de origen",
    ids,
    index=ids.index("IR01") if "IR01" in ids else 0,
    format_func=lambda codigo: f"{codigo} · {documentos.at[codigo, 'title']}",
)

st.write(documentos.at[seleccion, "text"])

resultados = motor.recommend(seleccion).rename(columns={
    "id": "ID",
    "title": "Documento",
    "reason": "Motivo",
    "similarity": "Similitud textual",
})

st.dataframe(
    resultados[["ID", "Documento", "Motivo", "Similitud textual"]],
    hide_index=True,
    column_config={
        "Similitud textual": st.column_config.NumberColumn(format="%.3f"),
    },
)

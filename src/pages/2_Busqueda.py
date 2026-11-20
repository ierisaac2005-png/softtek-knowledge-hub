import streamlit as st

from kms_utils import load_documents
from semantic_search import MultilingualSearch


@st.cache_resource(show_spinner="Preparando el índice de documentos…")
def cargar_buscador():
    return MultilingualSearch(load_documents())


st.title("Búsqueda de documentos")

with st.form("busqueda"):
    pregunta = st.text_input(
        "¿Qué quieres consultar?",
        placeholder="¿Qué debo hacer para pedir días de vacaciones?",
    )
    buscar = st.form_submit_button("Buscar")

if buscar:
    if pregunta.strip():
        resultados = cargar_buscador().search(pregunta, top_k=3)
        for posicion, doc in enumerate(resultados.itertuples(), start=1):
            st.subheader(f"{posicion}. {doc.id} · {doc.title}")
            st.caption(f"{doc.category} · Similitud: {doc.similarity:.3f}")
            st.write(doc.text)
    else:
        st.warning("Escribe una pregunta para buscar.")

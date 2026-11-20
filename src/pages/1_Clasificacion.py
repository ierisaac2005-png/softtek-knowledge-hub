import streamlit as st

from classifier import build_combined_pipeline, classify, review_status
from kms_utils import load_documents


@st.cache_resource
def cargar_clasificador():
    documentos = load_documents()
    return build_combined_pipeline().fit(
        documentos["content"], documentos["category"]
    )


st.title("Clasificación de documentos")

with st.form("clasificacion"):
    texto = st.text_area("Texto del documento en inglés", height=160)
    enviado = st.form_submit_button("Clasificar")

if enviado:
    if texto.strip():
        categoria, puntaje = classify(cargar_clasificador(), [texto.strip()])[0]
        st.subheader(categoria)
        st.write(f"Puntaje del modelo: {puntaje:.2%}")
        st.write(f"Estado: {review_status(puntaje)}")
    else:
        st.warning("Escribe o pega un documento antes de clasificar.")

import streamlit as st

from chatbot import KnowledgeChatbot


@st.cache_resource(show_spinner="Cargando los modelos del chatbot…")
def cargar_chatbot():
    return KnowledgeChatbot()


st.title("Chatbot de conocimiento")

with st.form("consulta_chatbot"):
    pregunta = st.text_input(
        "Pregunta sobre los documentos",
        placeholder="¿Con cuánta anticipación debo solicitar mis vacaciones?",
    )
    consultar = st.form_submit_button("Consultar")

if consultar:
    if pregunta.strip():
        chatbot = cargar_chatbot()
        with st.spinner("Consultando el documento y preparando la respuesta…"):
            resultado = chatbot.answer(pregunta)

        st.subheader("Respuesta")
        st.write(resultado["response"])
        st.caption(f"Fuente consultada: [{resultado['source_id']}] {resultado['title']}")
    else:
        st.warning("Escribe una pregunta para consultar.")

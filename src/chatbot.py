"""Chatbot de Knowledge Hub con recuperación de documentos y citas."""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kms_utils import load_documents
from semantic_search import MultilingualSearch


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
INSTRUCTIONS = (
    "Responde brevemente en español usando únicamente el documento. "
    "Si falta un dato necesario para responder, escribe exactamente: "
    "Información insuficiente."
)


EJEMPLOS_INTENCION = {'Ayuda del chatbot': ['Quiero saber qué puedes hacer por mí.',
                       'Explícame cómo usar este asistente.',
                       '¿Qué tipo de preguntas te puedo hacer?',
                       '¿Para qué sirve esta herramienta de conversación?'],
 'Consulta documental': ['¿Dónde presento una solicitud de vacaciones?',
                         '¿Quién supervisa el cumplimiento de la política de '
                         'privacidad?',
                         '¿Qué permisos se necesitan para acceder a los '
                         'sistemas?',
                         '¿Cuál fue la causa del retraso en el procesamiento '
                         'de pedidos?']}
UMBRAL_AYUDA = 0.020
RESPUESTA_AYUDA = (
    "Puedo buscar información en los documentos del proyecto, responder en español "
    "y mostrarte la fuente consultada. Puedes preguntarme cómo solicitar vacaciones "
    "o qué causó el incidente de pagos."
)


def pide_documento(pregunta):
    # Solo se activa cuando la petición comienza pidiendo un documento.
    patron = (
        r"^[¿¡\s]*(?:por favor[,\s]+)?"
        r"(?:(?:me (?:ayudas|puedes ayudar) a|puedes|quiero|necesito)\s+)?"
        r"(?:encontrar|buscar|localizar|ver|m[oó]strame|mu[eé]strame|busca|encuentra|localiza)"
        r"\s+(?:el|la|un|una)\s+(?:reporte|informe|documento|guía|guia|política|politica)\b"
    )
    return re.search(patron, pregunta, flags=re.IGNORECASE) is not None


class KnowledgeChatbot:
    def __init__(self):
        self.search = MultilingualSearch(load_documents())
        self.vectores_intencion = {
            ruta: self.search.model.encode(
                ["query: " + ejemplo for ejemplo in ejemplos],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for ruta, ejemplos in EJEMPLOS_INTENCION.items()
        }
        self.model = None

    def es_ayuda(self, pregunta):
        vector = self.search.model.encode(
            "query: " + pregunta, normalize_embeddings=True, show_progress_bar=False
        )
        puntajes = {}
        for ruta, vectores in self.vectores_intencion.items():
            mejores = sorted(float(valor) for valor in vectores @ vector)[-2:]
            puntajes[ruta] = sum(mejores) / 2
        return puntajes["Ayuda del chatbot"] - puntajes["Consulta documental"] >= UMBRAL_AYUDA

    def cargar_generador(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype="auto"
        ).eval()

    def answer(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Escribe una pregunta.")

        if self.es_ayuda(question):
            return {"response": RESPUESTA_AYUDA, "source_id": None, "title": None}

        source = self.search.search(question, top_k=1).iloc[0]
        if pide_documento(question):
            return {
                "response": f"Documento más cercano: {source['title']} [{source['id']}]",
                "source_id": source["id"],
                "title": source["title"],
            }

        if self.model is None:
            self.cargar_generador()
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": f"Documento:\n{source['text']}\n\nPregunta:\n{question}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=60, do_sample=False
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        if not text or text.casefold().rstrip(".") == "información insuficiente":
            response = "Información insuficiente."
        else:
            response = f"{text} [{source['id']}]"

        return {
            "response": response,
            "source_id": source["id"],
            "title": source["title"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="Preguntas entre comillas")
    args = parser.parse_args()

    if any(not question.strip() for question in args.questions):
        parser.error("Escribe una pregunta.")

    bot = KnowledgeChatbot()

    def show(question):
        result = bot.answer(question)
        print(f"\nPregunta: {question}")
        if result["source_id"]:
            print(f"Fuente consultada: [{result['source_id']}] {result['title']}")
        print(f"Respuesta: {result['response']}")

    if args.questions:
        for question in args.questions:
            show(question)
    else:
        print("Knowledge Hub · Escribe una pregunta o presiona Enter para salir.")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question:
                break
            show(question)


if __name__ == "__main__":
    main()

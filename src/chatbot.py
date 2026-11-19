"""Chatbot de Knowledge Hub con recuperación de documentos y citas."""

import argparse

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


class KnowledgeChatbot:
    def __init__(self):
        self.search = MultilingualSearch(load_documents())
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype="auto"
        ).eval()

    def answer(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Escribe una pregunta.")

        source = self.search.search(question, top_k=1).iloc[0]
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

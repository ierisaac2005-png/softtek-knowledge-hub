from collections import defaultdict

from sentence_transformers import SentenceTransformer


UMBRAL_AYUDA = 0.020

EJEMPLOS = {
    "Ayuda del chatbot": [
        "Quiero saber qué puedes hacer por mí.",
        "Explícame cómo usar este asistente.",
        "¿Qué tipo de preguntas te puedo hacer?",
        "¿Para qué sirve esta herramienta de conversación?",
    ],
    "Consulta documental": [
        "¿Dónde presento una solicitud de vacaciones?",
        "¿Quién supervisa el cumplimiento de la política de privacidad?",
        "¿Qué permisos se necesitan para acceder a los sistemas?",
        "¿Cuál fue la causa del retraso en el procesamiento de pedidos?",
    ],
}

PRUEBAS = [
    ("Ayuda del chatbot", "¿Cuáles son tus funciones?"),
    ("Ayuda del chatbot", "¿Cuáles son las funciones principales de este chat bot?"),
    ("Consulta documental", "¿Cuáles son las funciones del responsable de protección de datos?"),
    ("Consulta documental", "¿Con cuánta anticipación debo pedir días libres?"),
    ("Consulta documental", "¿Me puedes explicar por qué se acumularon los pedidos en la cola?"),
]


def evaluar(pregunta):
    consulta = modelo.encode("query: " + pregunta, normalize_embeddings=True)
    similitudes = vectores @ consulta

    por_ruta = defaultdict(list)
    for ruta, puntaje in zip(rutas, similitudes):
        por_ruta[ruta].append(float(puntaje))

    # Cada ruta recibe el promedio de sus dos ejemplos más cercanos.
    puntajes = {
        ruta: sum(sorted(valores, reverse=True)[:2]) / 2
        for ruta, valores in por_ruta.items()
    }

    ayuda = puntajes["Ayuda del chatbot"]
    documental = puntajes["Consulta documental"]
    ventaja_ayuda = ayuda - documental

    print(
        f"  Ayuda: {ayuda:.3f} | "
        f"Documental: {documental:.3f} | "
        f"Ventaja de ayuda: {ventaja_ayuda:+.3f}"
    )

    if ventaja_ayuda >= UMBRAL_AYUDA:
        return "Ayuda del chatbot"
    return "Consulta documental"


print("Cargando E5 en CPU…")
modelo = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

rutas = []
textos = []
for ruta, ejemplos in EJEMPLOS.items():
    for ejemplo in ejemplos:
        rutas.append(ruta)
        textos.append("query: " + ejemplo)

vectores = modelo.encode(textos, normalize_embeddings=True)

aciertos = 0
for esperada, pregunta in PRUEBAS:
    print(f"\n{pregunta}")
    elegida = evaluar(pregunta)
    print(f"  Esperada: {esperada} | Elegida: {elegida}")
    aciertos += elegida == esperada

print(f"\nAciertos en estas cinco pruebas: {aciertos}/{len(PRUEBAS)}")
print("\nEscribe tus propias preguntas. Presiona Enter sin texto para salir.")

while True:
    pregunta = input("\nPregunta: ").strip()
    if not pregunta:
        break
    print("  Ruta elegida:", evaluar(pregunta))

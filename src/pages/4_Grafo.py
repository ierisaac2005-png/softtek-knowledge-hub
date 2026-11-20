import json
from textwrap import fill

import streamlit as st
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDFS

from kms_utils import ONTOLOGY_PATH, load_documents


KMS = Namespace("http://softtek.example/kms#")


@st.cache_resource
def cargar_datos():
    grafo = Graph().parse(ONTOLOGY_PATH, format="turtle")
    documentos = load_documents().set_index("id")
    return grafo, documentos


def relaciones_documento(grafo, origen):
    relaciones = []

    for destino in grafo.objects(origen, KMS.references):
        relaciones.append((origen, "hace referencia a", destino))

    for incidente in grafo.objects(origen, KMS.documents):
        relaciones.append((origen, "documenta", incidente))

        for leccion in grafo.objects(incidente, KMS.resultsIn):
            relaciones.append((incidente, "da lugar a", leccion))

            for destino in grafo.objects(leccion, KMS.updates):
                relaciones.append((leccion, "actualiza", destino))

    return relaciones


def dibujar(grafo, origen, relaciones):
    nodos = {}
    lineas = [
        "digraph conocimiento {",
        'graph [rankdir=TB, bgcolor="#ffffff", pad="0.25"];',
        'node [shape=box, style="rounded,filled", color="#526578", '
        'fontcolor="#172536", fontname="Arial"];',
        'edge [color="#526578", fontcolor="#26384a", fontname="Arial"];',
    ]

    for desde, motivo, hasta in relaciones:
        for nodo in (desde, hasta):
            if nodo in nodos:
                continue

            clave = f"n{len(nodos)}"
            nodos[nodo] = clave
            titulo = grafo.value(nodo, RDFS.label) or str(nodo).split("#")[-1]
            etiqueta = fill(str(titulo), width=34)
            codigo = grafo.value(nodo, KMS.docId)
            if codigo:
                etiqueta = f"{codigo}\n{etiqueta}"

            color = "#dbeafe" if nodo == origen else "#dcfce7" if codigo else "#fff1cd"
            rotulo = json.dumps(etiqueta, ensure_ascii=False)
            lineas.append(f'{clave} [label={rotulo}, fillcolor="{color}"];')

        lineas.append(f'{nodos[desde]} -> {nodos[hasta]} [label="{motivo}"];')

    lineas.append("}")
    st.graphviz_chart("\n".join(lineas), width="stretch")


st.title("Grafo conceptual")

grafo, documentos = cargar_datos()
ids = documentos.index.tolist()
seleccion = st.selectbox(
    "Documento de origen",
    ids,
    index=ids.index("IR03") if "IR03" in ids else 0,
    format_func=lambda codigo: f"{codigo} · {documentos.at[codigo, 'title']}",
)

origen = grafo.value(predicate=KMS.docId, object=Literal(seleccion))
relaciones = relaciones_documento(grafo, origen) if origen is not None else []

if relaciones:
    dibujar(grafo, origen, relaciones)
else:
    st.info(
        "Este documento carece de vínculos de salida en las rutas evaluadas. "
        "Puedes consultar sus recomendaciones por similitud en la página principal."
    )

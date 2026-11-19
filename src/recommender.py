"""Recommend related Knowledge Hub documents using graph links and TF-IDF."""
import argparse

from rdflib import Graph, Literal, Namespace
from sklearn.metrics.pairwise import cosine_similarity

from kms_utils import ONTOLOGY_PATH, load_documents
from semantic_search import SemanticSearch


class DocumentRecommender:
    def __init__(self):
        self.search = SemanticSearch(load_documents())
        self.docs = self.search.df
        self.graph = Graph().parse(ONTOLOGY_PATH, format="turtle")
        ex = Namespace("http://softtek.example/kms#")
        self.doc_id = ex.docId
        self.routes = [
            (ex.references, "Referencia directa"),
            (ex.documents / ex.resultsIn / ex.updates, "Lección aprendida"),
        ]

    def recommend(self, source_id, k=5):
        """Return up to k distinct related documents, excluding the source."""
        if k < 1:
            raise ValueError("k debe ser mayor que cero")
        source_id = source_id.strip().upper()
        matches = self.docs.index[self.docs["id"] == source_id]
        if matches.empty:
            raise ValueError(f"Documento desconocido: {source_id}")

        matrix = self.search.doc_matrix
        scores = cosine_similarity(matrix[matches[0]], matrix)[0]
        ranking = self.docs[["id", "title", "category"]].assign(similarity=scores)
        ranking = ranking.loc[ranking["id"] != source_id].copy()

        links = {}
        for priority, (route, reason) in enumerate(self.routes):
            for source in self.graph.subjects(self.doc_id, Literal(source_id)):
                for target in self.graph.objects(source, route):
                    for target_id in self.graph.objects(target, self.doc_id):
                        links.setdefault(str(target_id), (priority, reason))

        fallback = (len(self.routes), "Similitud textual")
        evidence = [links.get(key, fallback) for key in ranking["id"]]
        ranking["priority"] = [priority for priority, _ in evidence]
        ranking["reason"] = [reason for _, reason in evidence]
        return (ranking.sort_values(["priority", "similarity", "id"],
                                    ascending=[True, False, True])
                .head(k).drop(columns="priority").reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", help="ID del documento, por ejemplo IR01")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    try:
        results = DocumentRecommender().recommend(args.source_id, args.k)
    except ValueError as error:
        parser.error(str(error))
    print(results.to_string(index=False, formatters={"similarity": "{:.3f}".format}))


if __name__ == "__main__":
    main()

"""Module to implement the base method of the MRI model"""
from corpus import CorpusAnalyzer
from typing import List, Tuple
from tools import Document


class MRI:
    def __init__(self, doc_analyzer: CorpusAnalyzer):
        # en un futuro debería elegir el corpus parser dependiendo en el path
        self.doc_analyzer = doc_analyzer

    def ranking_function(self, query: List[Tuple[int, int]]) -> List[Tuple[int, float]]:
        """
        Main function that returns a sorted ranking of the similarity
        between the corpus and the query.
        format: [doc_id, similarity]
        """
        raise NotImplementedError

    def get_similarity_docs(self, ranking: List[Tuple[int, float]]) -> List[Document]:
        """
        Uses the ranking produced by the ranking function
        and returns the documents with the highest ranking.
        """
        return [self.doc_analyzer.id2doc(doc_id) for doc_id, _ in ranking]

from models.model import MRI
from typing import Dict, List, Tuple
from utils import tf, idf
import math


class VectorMRI(MRI):
    def __init__(self, doc_analyzer):
        MRI.__init__(self, doc_analyzer)
        # parameters in the query weights
        self.a = 0.4  # 0.5

    def ranking_function(self, query: List[Tuple[int, int]]) -> List[Tuple[int, float]]:
        """
        Main function that returns a sorted ranking of the similarity
        between the corpus and the query.
        format: [doc_id, similarity]
        """
        ranking = []
        query_vect = dict(query)
        for i, doc in enumerate(self.doc_analyzer.documents):
            num = 0
            doc_weights_sqr = 0
            query_weights_sqr = 0
            for ti in query_vect.keys():
                w_doc = self.weight_doc(ti, i)
                w_query = self.weight_query(ti, query_vect)
                num += w_doc * w_query
                doc_weights_sqr += w_doc ** 2
                query_weights_sqr += w_query ** 2
            try:
                sim = num / (math.sqrt(doc_weights_sqr) * math.sqrt(query_weights_sqr))
            except ZeroDivisionError:
                sim = 0
            if sim > 0.3:
                ranking.append((doc.id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def weight_query(self, ti: int, query_vect: Dict[int, int]):
        freq = query_vect[ti]
        max_freq = max(query_vect.values())
        tf = freq / max_freq
        idf = self.idf(ti)
        return (self.a + (1 - self.a) * tf) * idf

    def weight_doc(self, ti: int, dj: int) -> float:
        return self.tf(ti, dj) * self.idf(ti)

    def tf(self, ti: int, dj: int) -> float:
        return tf(self.doc_analyzer, ti, dj)

    def idf(self, ti: int) -> float:
        return idf(self.doc_analyzer, ti)

from models.model import MRI
from typing import Dict
import math

class VectorMRI(MRI):
    def __init__(self, doc_analyzer):
        MRI.__init__(self, doc_analyzer)
        # parameters in the query weights
        self.a = 0.4  # 0.5

    def ranking_function(self, query: str):
        ranking = []
        query_vect = dict(self.query_parser(query, self.doc_analyzer.index))
        for i, doc in enumerate(self.doc_analyzer.documents):
            num = 0
            doc_weights_sqr = 0
            query_weights_sqr = 0
            for ti in query_vect.keys():
                w_doc = self.weight_doc(ti, i)
                w_query = self.weight_query(ti, query_vect)
                num += w_doc * w_query
                doc_weights_sqr += w_doc**2
                query_weights_sqr += w_query**2
            try:
                sim = num / math.sqrt(doc_weights_sqr) * math.sqrt(query_weights_sqr)
            except ZeroDivisionError:
                sim = 0
            ranking.append((doc.id, sim))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def weight_query(self, ti: int, query_vect: Dict[int, int]):
        freq = query_vect[ti]
        max_freq = max(query_vect.values())
        tf = freq / max_freq
        idf = self.idf(ti)
        return (self.a + (1 - self.a) * tf) * idf

    def weight_doc(self, ti: int, dj: int) -> int:
        return self.tf(ti, dj) * self.idf(ti)

    def tf(self, ti: int, dj: int) -> int:
        freq = self.doc_analyzer.get_frequency(ti, dj)
        max_freq_tok, max_freq = self.doc_analyzer.get_max_frequency(dj)
        return freq / max_freq

    def idf(self, ti: int):
        N = len(self.doc_analyzer.documents)
        ni = self.doc_analyzer.index.dfs[ti]
        return math.log2(N / ni)
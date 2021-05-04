"""Retroalimentation of the ir system to obtain a better query"""
from typing import List, Tuple, Dict
from corpus import CorpusAnalyzer
from query import QueryParser


class RocchioAlgorithm:
    def __init__(self, query: str, corpus: CorpusAnalyzer, relevant_docs: List[int], non_relevant_docs: List[int]):
        """
        Initializes the rocchio algorithm, relevance is a list of tuples (doc_id, relevance)
        """
        self.corpus = corpus
        self.rel_docs = relevant_docs
        self.non_rel_docs = non_relevant_docs
        self.query_vect = QueryParser().get_query_vector(query, corpus.index)

    def __call__(self, *, alpha=1, beta=0.75, gamma=0.15):
        term1 = {ti: alpha * q for ti, q in self.query_vect}

        sum_rel_docs = self._sum_docs_vect(self.rel_docs)
        n = len(self.rel_docs)
        term2 = {ti: (beta / n) * v for ti, v in sum_rel_docs.items()}

        sum_non_rel_docs = self._sum_docs_vect(self.non_rel_docs)
        n = len(self.non_rel_docs)
        term3 = {ti: -(gamma / n) * v for ti, v in sum_non_rel_docs.items()}

        new_query = term1
        new_query = self._sum_2_vect(new_query, term2)
        new_query = self._sum_2_vect(new_query, term3)
        return list(new_query.items())

    def _sum_docs_vect(self, docs):
        sum_docs = {}
        for doc_id in docs:
            for term_id, freq in self.corpus.doc2bow(doc_id):
                try:
                    sum_docs[term_id] += freq
                except KeyError:
                    sum_docs[term_id] = freq
        return sum_docs

    def _sum_2_vect(self, doc_vect1: Dict[int, int], doc_vect2: Dict[int, int]) -> Dict[int, int]:
        # note: the result is stored in doc_vect1 for efficiency
        for ti, freq in doc_vect2.items():
            try:
                doc_vect1[ti] += freq
            except KeyError:
                doc_vect1[ti] = freq
        return doc_vect1

    def get_tokens_by_vector(self, query_vect: List[Tuple[int, int]], n=10) -> List[str]:
        """Gets an approximation of the new query tokens"""
        sorted_frq = sorted(query_vect, key=lambda x: x[1], reverse=True)
        return [self.corpus.index[tok_id] for tok_id, _ in sorted_frq[:n]]
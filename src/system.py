"""Main class that encapsulates all system functionalities"""
from corpus import CorpusAnalyzer
from query import QueryParser
from models import MRI
from feedback import RocchioAlgorithm
from typing import List, Tuple
from tools import Document
from query_expansion import query_expansion_with_nltk

# TODO: Save the weights for every document in the collection for more efficiency
# TODO: Save every query stored with the Rocchio algorithm


class IRSystem:
    def __init__(self, model: MRI, corpus: CorpusAnalyzer):
        self.model = model
        self.corpus = corpus

    def make_query(self, query: str) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        ranking = self.model.ranking_function(query)
        # maybe should return the ranking for relevance feedback
        return self.model.get_similarity_docs(ranking)

    def user_feedback(self, query: str, relevant_docs, total_docs):
        """
        Feedback if the user helped
        :param query: The initial query of the user
        :param relevance: The list of the documents id the user found relevant
        :param total_docs: The total list of documents id that were showed to the user
        :return: The vector of the new query
        """
        non_relevant_docs = [doc_id for doc_id in total_docs if doc_id not in relevant_docs]
        rocchio = RocchioAlgorithm(query, self.corpus, relevant_docs, non_relevant_docs)
        new_query_vect = rocchio()
        new_query_tok = rocchio.get_tokens_by_vector(new_query_vect)
        # No tengo claro que hacer con este vector, posiblemente haga un diccionario estático para guardarlo
        return new_query_vect

    def pseudo_feedback(self, query: str, ranking: List[Tuple[int, float]], k=10):
        """
        To use if the user didn't helped in the feedback.
        The k-highest ranked documents are selected as relevants and the feedback starts.
        """
        relevant_docs = [doc_id for doc_id, _ in ranking[:k]]
        non_relevant_docs = [doc_id for doc_id, _ in ranking[k:]]
        rocchio = RocchioAlgorithm(query, self.corpus, relevant_docs, non_relevant_docs)
        new_query_vect = rocchio()
        new_query_tok = rocchio.get_tokens_by_vector(new_query_vect)
        # No tengo claro que hacer con este vector, posiblemente haga un diccionario estático para guardarlo
        return new_query_vect

    @staticmethod
    def global_query_expansion(query: str):
        """Does global query expansion, returning a list of possible queries related with the original one"""
        query_tokens = QueryParser().parse(query)
        return query_expansion_with_nltk(query_tokens)

"""Main class that encapsulates all system functionalities"""
from corpus import CorpusAnalyzer
from query import QueryParser
from models import MRI
from feedback import RocchioAlgorithm
from typing import List, Tuple, Iterable
from tools import Document
from query_expansion import query_expansion_with_nltk
from clustering import ClusterManager

# TODO: Save the weights for every document in the collection for more efficiency


class IRSystem:
    def __init__(self, model: MRI, corpus: CorpusAnalyzer):
        self.model = model
        self.corpus = corpus
        if corpus.name != 'union':
            # with the union of the datasets, the cluster algorithm cannot run
            self.clusterer = ClusterManager(corpus)
            self.clusterer.fit_cluster(8)
        else:
            self.clusterer = None
        self.query_parser = QueryParser()

    def make_query(self, query: str) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        query_vect = self.query_parser(query, self.corpus.index)

        # Tries to find another vector with the feedback model
        new_query_vect = RocchioAlgorithm.load_query(query)
        query_vect = query_vect if new_query_vect is None else new_query_vect

        ranking = self.model.ranking_function(query_vect)
        docs = self.model.get_similarity_docs(ranking)

        # Doing clustering to return related documents with the one with the highest score
        if self.clusterer is not None:
            related_docs = self.clusterer.get_cluster_samples(docs[0].id)
            related_docs = [self.corpus.id2doc(doc_id) for doc_id in related_docs[:10]]
            docs = set(docs[:20]).union(set(related_docs[:10])).union(docs[20:])
        # maybe should return the ranking for relevance feedback
        return list(docs)

    def user_feedback(self, query: str, relevant_docs: List[int], total_docs: List[int]):
        """
        Feedback if the user helped
        :param query: The initial query of the user
        :param relevant_docs: The list of the documents id the user found relevant
        :param total_docs: The total list of documents id that were showed to the user
        :return: The vector of the new query
        """
        non_relevant_docs = [doc_id for doc_id in total_docs if doc_id not in relevant_docs]
        rocchio = RocchioAlgorithm(query, self.corpus, relevant_docs, non_relevant_docs)
        new_query_vect = rocchio()
        new_query_tok = rocchio.get_tokens_by_vector(new_query_vect)
        # No tengo claro que hacer con este vector, posiblemente haga un diccionario estático para guardarlo
        rocchio.save_query(query, new_query_vect)
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
        rocchio.save_query(query, new_query_vect)
        return new_query_vect

    @staticmethod
    def global_query_expansion(query: str):
        """Does global query expansion, returning a list of possible queries related with the original one"""
        query_tokens = QueryParser().parse(query)
        return query_expansion_with_nltk(query_tokens)

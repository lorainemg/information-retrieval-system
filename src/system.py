"""Main class that encapsulates all system functionalities"""
from corpus import CorpusAnalyzer
from query import QueryParser
from models import MRI
from feedback import RocchioAlgorithm
from typing import List, Tuple, Iterable
from tools import Document
from query_expansion import query_expansion
from clustering import ClusterManager
from document_recommendation import DocumentRecommender

# TODO: Save the weights for every document in the collection for more efficiency


class IRSystem:
    def __init__(self, model: MRI):
        self.model = model
        self.corpus = model.doc_analyzer
        stemming = self.corpus.stemmer is not None
        if self.corpus.name != 'union':
            # with the union of the datasets, the cluster algorithm cannot run
            self.clusterer = ClusterManager(self.corpus)
            self.clusterer.fit_cluster(4)
        else:
            self.clusterer = None
        self.query_parser = QueryParser(stemming)
        self.document_recommender = DocumentRecommender(self.clusterer, self.corpus)

    def make_query(self, query: str, ranking=False) -> List[Document]:
        """Makes a query with the loaded corpus and returns the documents sorted for relevancy"""
        query_vect = self.query_parser(query, self.corpus.index)

        # Tries to find another vector with the feedback model
        new_query_vect = self._load_query(query)
        query_vect = query_vect if new_query_vect is None else new_query_vect

        doc_ranking = self.model.ranking_function(query_vect)
        docs = self.model.get_similarity_docs(doc_ranking)

        # Doing clustering to return related documents with the one with the highest score
        if self.clusterer is not None:
            try:
                related_docs = self.clusterer.get_cluster_samples(self.corpus.mapping[docs[0].id])
                related_docs = [self.corpus.documents[doc_id] for doc_id in related_docs[:10]]
                docs = docs[:20] + [d for d in related_docs[:10] if d not in docs[:20]]
            except:
                pass

        # The most similar doc to the query is saved as relevant to the user for the document recommender
        if len(docs) > 0:
            self.document_recommender.add_rating(docs[0].id, 1)
        if ranking:
            return docs, doc_ranking
        else:
            return docs

    def _load_query(self, query: str):
        """Tries to find another vector for the query with the feedback model"""
        try:
            return RocchioAlgorithm.load_query(query, self.corpus.name)
        except FileNotFoundError:
            return None

    def user_feedback(self, query: str, relevant_docs: List[int], total_docs: List[int]):
        """
        Feedback if the user helped
        :param query: The initial query of the user
        :param relevant_docs: The list of the documents id the user found relevant
        :param total_docs: The total list of documents id that were showed to the user
        :return: The vector of the new query
        """
        non_relevant_docs = [doc_id for doc_id in total_docs if doc_id not in relevant_docs]

        # For the recommender system: relevants docs are stored as interesting, non relevants as none
        self.document_recommender.add_ratings({doc_id: 1 for doc_id in relevant_docs})

        # Execution of the Rocchio Algorithm
        rocchio = RocchioAlgorithm(query, self.corpus, relevant_docs, non_relevant_docs)
        new_query_vect = rocchio()
        rocchio.save_query(query, new_query_vect, self.corpus.name)
        return new_query_vect

    def pseudo_feedback(self, query: str, ranking: List[Tuple[int, float]], k=10):
        """
        To use if the user didn't helped in the feedback.
        The k-highest ranked documents are selected as relevants and the feedback starts.
        """
        relevant_docs = [doc_id for doc_id, _ in ranking[:k]]
        non_relevant_docs = [doc_id for doc_id, _ in ranking[k:]]

        # For the recommender system: relevants docs are stored as interesting, non relevants as none
        self.document_recommender.add_ratings({doc_id: 1 for doc_id in relevant_docs})

        # Execution of the Rocchio Algorithm
        rocchio = RocchioAlgorithm(query, self.corpus, relevant_docs, non_relevant_docs)
        new_query_vect = rocchio()
        rocchio.save_query(query, new_query_vect, self.corpus.name)
        return new_query_vect

    def global_query_expansion(self, query: str) -> List[str]:
        """Does global query expansion, returning a list of possible queries related with the original one"""
        query_tokens = QueryParser().parse(query)
        return query_expansion(query_tokens, self.corpus)

    def get_recommended_documents(self) -> List[Document]:
        """
        Returns the 5 more interesting useen documents for the user
        according to the recommendation system
        """
        # If there is no rated document return zero
        if len(self.document_recommender.ratings) == 0:
            return []
        docs_id = self.document_recommender.recommend_documents(5)
        return [self.corpus.documents[doc_id] for doc_id in docs_id]

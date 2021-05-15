from corpus import CorpusAnalyzer
from clustering import ClusterManager
from typing import Dict
from pathlib import Path
import numpy as np
import json
import os


class DocumentRecommender:
    def __init__(self, clusterer: ClusterManager, corpus: CorpusAnalyzer, ratings: Dict[int, int]=None):
        """
        Document Recommender initialization
        :param clusterer: corpus with the documents
        :param ratings: ratings of the documents consisting in: [doc_id, rating]
        (usually a rating of 0 or 1 if the user found the document interesting)
        """
        self.corpus: CorpusAnalyzer = corpus
        self.clusterer: ClusterManager = clusterer
        self.name = self.corpus.name
        if ratings is None:
            # the ratings are obtained in the ratings folder
            self.load_ratings()
        else:
            self.ratings = ratings

    @property
    def mean_of_items(self):
        return sum(self.ratings.values()) / len(self.ratings)

    def similarity(self, doc_i: int, doc_j: int) -> float:
        """Returns the similarity between two documents"""
        return self.jaccard_distance(doc_i, doc_j)
        # return self.cosine_distance(doc_i, doc_j)

    def jaccard_distance(self, doc_i, doc_j):
        """Binary distance between 2 documents"""
        # vectors cotaining the terms of the documents
        vect_i = set(self.corpus.doc2bow(doc_i).keys())
        vect_j = set(self.corpus.doc2bow(doc_j).keys())
        intersect = len(vect_i.intersection(vect_j))
        union = len(vect_i.union(vect_j))
        return intersect / union

    def cosine_distance(self, doc_i, doc_j):
        vect_i = self.corpus.doc2bow(doc_i)
        vect_j = self.corpus.doc2bow(doc_j)
        dot_product = 0
        for term, freq in vect_i.items():
            try:
                dot_product += vect_j[term]*freq
            except KeyError:
                pass
        vect_i_l2norm = np.sqrt(sum(map(lambda x: x**2, vect_i.values())))
        vect_j_l2norm = np.sqrt(sum(map(lambda x: x**2, vect_j.values())))
        return dot_product / (vect_i_l2norm * vect_j_l2norm)

    def add_rating(self, doc_id: int, rating: int):
        """Adds a rating for the doc_id in the ratings list"""
        self.ratings[doc_id] = rating
        self.save_ratings()

    def add_ratings(self, ratings: Dict[int, int]):
        """Adds the rating for different documents"""
        self.ratings.update(ratings)
        self.save_ratings()

    def doc_deviation(self, doc_id: int):
        """Mean deviation of a document"""
        try:
            return self.ratings[doc_id] - self.mean_of_items
        except KeyError:
            # The doc_id is not in the ratings, therefore its value is zero
            return -self.mean_of_items

    def predictor_baseline(self, doc_id: int):
        return self.mean_of_items + self.doc_deviation(doc_id)

    def expected_rating(self, doc_id: int):
        """Predicts the rating of an unseen document"""
        if self.clusterer is not None:
            documents = self.clusterer.get_cluster_samples(doc_id)
        else:
            documents = [doc_id for doc_id in range(len(self.corpus.documents))]
        rated_documents = [doc_id for doc_id in documents if doc_id in self.ratings]
        numerator = sum(map(lambda d: self.similarity(doc_id, d)*(self.ratings[d] - self.predictor_baseline(d)), rated_documents))
        denominator = sum(map(lambda d: self.similarity(doc_id, d), rated_documents))
        try:
            return self.predictor_baseline(doc_id) + numerator / denominator
        except ZeroDivisionError:
            return 0

    def recommend_documents(self, k=5):
        """Searches through all the documents and returns the best `k` recommendations"""
        doc_ratings = {}
        for doc_id in range(len(self.corpus.documents)):
            if doc_id not in self.ratings:
                predicted_rating = self.expected_rating(doc_id)
                doc_ratings[doc_id] = predicted_rating
        return sorted(doc_ratings, key=lambda x: doc_ratings[x])[:k]

    def load_ratings(self):
        try:
            self.ratings = json.load(open(Path(f'../resources/ratings/{self.name}_ratings.json'), 'r'))
        except FileNotFoundError:
            self.ratings = {}

    def save_ratings(self):
        try:
            json.dump(self.ratings, open(Path(f'../resources/ratings/{self.name}_ratings.json'), 'w'))
        except FileNotFoundError:
            json.dump(self.ratings, open(Path(f'../resources/ratings/{self.name}_ratings.json'), 'x'))
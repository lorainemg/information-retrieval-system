"""Module to handle clustering of the corpus for better performance"""
from sklearn.cluster import KMeans
from corpus import CorpusAnalyzer
from yellowbrick.cluster import KElbowVisualizer
from utils import tf, idf
import numpy as np
import pandas as pd
import pickle


class ClusterManager:
    def __init__(self, corpus: CorpusAnalyzer):
        self.corpus = corpus
        # load is used to load a saved model, used for efficiency
        try:
            self.load_model()
        except FileNotFoundError or FileExistsError:
            self.X = self.create_doc_vectors()
            self.cluster_map = pd.DataFrame()
            self.model = KMeans()

    def create_doc_vectors(self):
        """
        Creates the training examples of the clusterer creating vectors
        of the documents in the set.
        """
        examples = []
        for doc_id in range(self.corpus.index.num_docs):
            examples.append(self.get_doc_vector(doc_id))
        return np.array(examples)

    def get_doc_vector(self, doc_id: int):
        """Gets the vector of a single document"""
        bow = self.corpus.doc2bow(doc_id)
        tf_idf_vectors = np.zeros(len(self.corpus.index))
        for term_id, freq in bow.items():
            tf_idf_vectors[term_id] = tf(self.corpus, term_id, doc_id) * idf(self.corpus, term_id)
        return tf_idf_vectors

    def elbow_method(self) -> int:
        """Gets the optimus k by the elbow method"""
        visualizer = KElbowVisualizer(self.model, k=(4, 20), metric='calinski_harabasz')
        # visualizer = KElbowVisualizer(KMeans(), k=(4, 20))
        visualizer.fit(self.X)
        visualizer.show()
        return visualizer.elbow_value_

    def fit_cluster(self, k: int, *, load=True):
        """
        Does the training of k-means.
        Also stores all the documents clusters.
        """
        if not load:
            self.fit_cluster(k)
            return
        try:
            self.load_model()
        except FileNotFoundError or FileExistsError:
            self._fit_cluster(k)

    def _fit_cluster(self, k: int):
        self.model = KMeans(n_clusters=k)
        km = self.model.fit(self.X)
        self.cluster_map['doc_id'] = list(range(self.X.shape[0]))
        self.cluster_map['cluster'] = km.labels_
        self.save_model()

    def predict_cluster(self, doc_id):
        """Predicts the cluster of a given doc_id"""
        doc_vec = self.get_doc_vector(doc_id)
        return self.model.predict(np.array([doc_vec]))[0]

    def get_cluster_samples(self, doc_id):
        """Gets of the samples that are in the same cluster as `doc_id`"""
        cluster = self.predict_cluster(doc_id)
        return self.cluster_map[self.cluster_map.cluster == cluster].doc_id.array

    def save_model(self):
        """Saves kmeans model and cluster map"""
        pickle.dump(self.model, open(f'../resources/cluster/{self.corpus.name}_kmeans.pkl', 'wb'))
        pickle.dump(self.cluster_map, open(f'../resources/cluster/{self.corpus.name}_cluster_map.pkl', 'wb'))

    def load_model(self):
        """Loads kmeans model and cluster map"""
        self.model = pickle.load(open(f'../resources/cluster/{self.corpus.name}_kmeans.pkl', 'rb'))
        self.cluster_map = pickle.load(open(f'../resources/cluster/{self.corpus.name}_cluster_map.pkl', 'rb'))

"""Module to implement the base method of the MRI model"""
from corpus import CorpusAnalyzer
from query import QueryParser


class MRI:
    def __init__(self, doc_analyzer: CorpusAnalyzer):
        # en un futuro debería elegir el corpus parser dependiendo en el path
        self.doc_analyzer = doc_analyzer
        self.query_parser = QueryParser()

    def ranking_function(self, query: str):
        raise NotImplementedError

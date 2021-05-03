"""Module to implement the base method of the MRI model"""
from corpus import CranCorpusAnalyzer

class MRI:
    def __init__(self, corpus_path):
        # en un futuro debería elegir el corpus parser dependiendo en el path
        doc_analyzer = CranCorpusAnalyzer(corpus_path)


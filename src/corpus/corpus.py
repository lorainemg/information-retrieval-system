"""Corpus to read and parse corpus"""
from typing import Dict, List
from tools import Document


class CorpusAnalyzer:
    def __init__(self, corpus_path: str):
        self.corpus_fd = open(corpus_path, 'r')
        # A dictionary of documents where the keys are their identifier
        # and the value its a Document instance
        self.documents: Dict[int or str, Document] = {}

    def parse_documents(self):
        """Parse the documents and fills the documents instance, tokenizing"""
        raise NotImplementedError

    @staticmethod
    def remove_punctuation(string: str) -> str:
        punctuation = ".:,!\"#$%&()*+/;<=>?@[\]^_`{|}~?"
        cleaned_str = "                                "
        transform = punctuation.maketrans(cleaned_str)
        return string.translate(transform)

    @staticmethod
    def convert_to_lower(string: str):
        return string.lower()

    @staticmethod
    def tokenize(string: str):
        return string.split()

"""Module to parse a query"""
import nltk
from typing import List
from gensim.corpora import Dictionary

from utils import convert_to_lower, remove_punctuation, tokenize


class QueryParser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()

    def parse(self, text: str, stemming=False):
        text = convert_to_lower(remove_punctuation(text))
        tokens = tokenize(text)
        if stemming:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def get_query_vector(self, text: str, index: Dictionary):
        """
        Builds the vector of a query based in the index dictionary
        format: list of (token_id, token_count) 2-tuples.
        """
        return index.doc2bow(self.parse(text))

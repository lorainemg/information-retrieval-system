"""Module to parse a query"""
import nltk
from typing import List
from gensim.corpora import Dictionary

from utils import convert_to_lower, remove_punctuation, tokenize


class QueryParser:
    def __init__(self, stemming=False):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        if stemming:
            self.stemmer = nltk.PorterStemmer()
        else:
            self.stemmer = None

    def parse(self, text: str):
        text = convert_to_lower(remove_punctuation(text))
        tokens = tokenize(text)
        tokens = [tok for tok in tokens if tok not in self.stopwords]
        # not sure if i should remove the stopwords in the query
        if self.stemmer is not None:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def get_query_vector(self, text: str or List[str], index: Dictionary):
        """
        Builds the vector of a query based in the index dictionary
        format: list of (token_id, token_count) 2-tuples.
        """
        if isinstance(text, str):
            return index.doc2bow(self.parse(text))
        else:
            return index.doc2bow(text)

    def __call__(self, text: str, index: Dictionary):
        return self.get_query_vector(text, index)

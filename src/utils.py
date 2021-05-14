import nltk
import math


def remove_punctuation(string: str) -> str:
    punctuation = ".:,!\"#$%&()*+/;<=>?@[\]^_`{|}~?"
    cleaned_str = "                               "
    transform = str.maketrans(punctuation, cleaned_str)
    return string.translate(transform)


def convert_to_lower(string: str):
    return string.lower()


def tokenize(string: str):
    return nltk.wordpunct_tokenize(string)


def tf(doc_analyzer: "CorpusAnalyzer", ti: int, dj: int) -> float:
    """tf of a term in a document"""
    freq = doc_analyzer.get_frequency(ti, dj)
    max_freq_tok, max_freq = doc_analyzer.get_max_frequency(dj)
    return freq / max_freq


def idf(doc_analyzer: "CorpusAnalyzer", ti: int) -> float:
    """idf of a term in a document"""
    N = len(doc_analyzer.documents)
    ni = doc_analyzer.index.dfs[ti]
    return math.log2(N / ni)

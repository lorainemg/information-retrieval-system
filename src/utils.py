import nltk
from gensim import corpora
from typing import List

def remove_punctuation(string: str) -> str:
    punctuation = ".:,!\"#$%&()*+/;<=>?@[\]^_`{|}~?"
    cleaned_str = "                               "
    transform = str.maketrans(punctuation, cleaned_str)
    return string.translate(transform)


def convert_to_lower(string: str):
    return string.lower()


def tokenize(string: str):
    return nltk.wordpunct_tokenize(string)

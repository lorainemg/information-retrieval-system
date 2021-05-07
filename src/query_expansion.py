"""Module to implement the query expansion"""
from nltk.corpus import wordnet as wn
from typing import List

from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams


def query_expansion_with_nltk(query: List[str]):
    """
    Does query expansion with nltk.
    Performs spell checking and returns synonyms
    """
    tokens = spell_checking(query)
    synonyms = get_synonyms(query)
    synonyms.insert(0, tokens)
    return synonyms


def spell_checking(tokens: List[str]) -> List[str]:
    """Returns the correct spelling of every word in the list of tokens"""
    correct_words = words.words()
    correct_spelling = []
    for token in tokens:
        temp = [(edit_distance(token, w), w) for w in correct_words if w[0] == token[0]]
        word = min(temp, key=lambda x: x[0])[1]
        correct_spelling.append(word)
    return correct_spelling


def get_synonyms(tokens: List[str]) -> List[List[str]]:
    """
    Gets all the possible combinations of all the tokens being
    substitute by their synonyms
    """
    # no funciona muy bien...
    synonyms = []
    for token in tokens:
        try:
            syns = wn.synsets(token)[0].lemma_names()
            synonyms.append(syns)
        except IndexError:
            pass
    return _get_combinations(synonyms, 5)


def _get_combinations(collection, n_comb):
    """In collection gives n_comb combinations"""
    def combinate(n, result, final_result):
        nonlocal n_comb
        if n_comb == 0:
            return
        elif n == len(collection):
            final_result.append(result)
            n_comb -= 1
        else:
            for i in range(len(collection[n])):
                result[n] = collection[n][i]
                combinate(n + 1, result, final_result)

    final_result = []
    combinate(0, [0] * len(collection), final_result)
    return final_result

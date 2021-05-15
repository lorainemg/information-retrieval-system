"""Module to implement the query expansion"""
from nltk.corpus import wordnet as wn
from typing import List, Dict
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from utils import idf
import math


def query_expansion(query: List[str], doc_analyzer, mode='hypernym') -> List[str]:
    """
    Does query expansion using WordNet hypernyms and synonyms.
    :param query: list of tokenized query terms
    :param mode: choose mode: hypernym for hypernym based relaxation, 'synonym' for synonym based relaxation
    :return: list containing a list of alternate tokenized queries
    """
    tokens, change = spell_checking(query)
    new_tokens = get_relaxed_query(query, doc_analyzer)
    if change:
        return [tokens] + new_tokens
    else:
        return new_tokens


def spell_checking(tokens: List[str]) -> str:
    """Returns the correct spelling of every word in the list of tokens"""
    correct_words = words.words()
    correct_spelling = []
    change = False
    for token in tokens:
        temp = [(edit_distance(token, w), w) for w in correct_words if w[0] == token[0]]
        word = min(temp, key=lambda x: x[0])[1]
        if word != token:
            change = True
        correct_spelling.append(word)
    return " ".join(correct_spelling), change


def get_relaxed_query(tokens: List[str], doc_analyzer) -> List[str]:
    """
    Gets all the possible combinations of all the tokens being
    substitute by their synonyms or hypernyms
    """
    threshold = math.log2(len(doc_analyzer.documents) / 10)
    alternatives = {}
    for i, token in enumerate(tokens):
        # Iterating through query term list to check if the term is present in the corpus and qualifies the threshold
        if token not in doc_analyzer.index.token2id or idf(doc_analyzer,
                                                           doc_analyzer.index.token2id[token]) > threshold:
            for element in wn.synsets(token):
                # hyp and syn lists corresponding to the hypernyms and synonyms of most common context
                try:
                    try:
                        alternatives[i].union(set(element.hypernyms()[0].lemma_names())).union(set(element.lemma_names()))
                    except KeyError:
                        alternatives[i] = set(element.hypernyms()[0].lemma_names()).union(element.lemma_names())
                except IndexError:
                    pass
    return get_alternative_tokens(tokens, threshold, alternatives, doc_analyzer)


def get_alternative_tokens(tokens: List[str], threshold: float, alternatives: Dict[int, set], doc_analyzer):
    alternative_tokens = []
    # Replacing the query term which crosses threshold with its hypernyms
    for idx, alternatives in alternatives.items():
        for word in alternatives:
            # creating a copy of the original tokenized query
            temp = tokens.copy()
            # finding the index to make the replacement
            term = temp[idx]
            if term is not word and term in doc_analyzer.index.token2id and \
                    idf(doc_analyzer, doc_analyzer.index.token2id[term]) <= threshold:
                # Making sure the original term and its hypernym are not the same
                temp[idx] = word
                # adding the relaxed query
                alternative_tokens.append(" ".join(temp))
    return alternative_tokens


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

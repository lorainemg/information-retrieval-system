"""Module to implement the evaluation metrics"""
from typing import List


def precision_score(relevant: List, recovered: List) -> float:
    """Precision score is: which of the documents marked as relevant are really relevant"""
    # Recovered relevant
    rr = [d for d in recovered if d in relevant]
    return len(rr) / len(recovered)


def recall_score(relevant: List, recovered: List) -> float:
    """Recall score is: which of the total relevant documents where recoverd"""
    # Recovered relevant
    rr = [d for d in recovered if d in relevant]
    return len(rr) / len(relevant)


def fbeta_score(relevant: List, recovered: List, beta: float) -> float:
    """Score that harmonize precision and recall"""
    p = precision_score(relevant, recovered)
    r = recall_score(relevant, recovered)
    try:
        return (1 + beta ** 2) / (1 / p + (beta ** 2) / r)
    except ZeroDivisionError:
        return 0


def f1_score(relevant: List, recovered: List) -> float:
    """Particular case of the fbeta_score"""
    return fbeta_score(relevant, recovered, 1)


def fallout(relevant: List, recovered: List, total: int) -> float:
    # Recovered no relevant
    ri = [d for d in recovered if d not in relevant]
    # Total of non relevants documents
    irrelevant = total - len(relevant)
    return len(ri) / irrelevant


# ---------------------------- R scores ---------------------------- #


def r_precision_score(r: int, relevant: List, recovered: List) -> float:
    """Precision for `r` relevants documents"""
    return precision_score(relevant, recovered[:r])


def r_recall_score(r: int, relevant: List, recovered: List) -> float:
    """Precision for `r` relevants documents"""
    return recall_score(relevant, recovered[:r])


def r_f1_score(r: int, relevant: List, recovered: List) -> float:
    """Precision for `r` relevants documents"""
    return f1_score(relevant, recovered[:r])


def r_fallout_score(r: int, relevant: List, recovered: List, total: int) -> float:
    """Precision for `r` relevants documents"""
    return fallout(relevant, recovered[:r], total)
"""Tests a IR Model with a given metric with the Cranfield documents"""
import re
import json
from models import MRI, VectorMRI
from typing import List, Dict, Tuple
from corpus import CranCorpusAnalyzer
from evaluation import f1_score


def parse_query(query_path: str) -> Dict[int, str]:
    """Parses the query Cranfield file"""
    query_fd = open(query_path, 'r+')

    queries = {}
    current_id: int = None
    current_lines = []
    id_re = re.compile(r'\.I (\d+)')
    # marca cuando empieza el texto del documento actual
    getting_words = False
    for line in query_fd.readlines():
        m = id_re.match(line)
        # se empieza una nueva query
        if m is not None:
            # había un documento actual que se guarda en la lista de documentos
            if len(current_lines) > 0:
                queries[current_id] = " ".join(current_lines)
            current_id = int(m.group(1))
            current_lines = []
            getting_words = False
        elif line.startswith('.W'):
            getting_words = True
        elif getting_words:
            current_lines.append(line)
    return queries


def parse_test_file(test_path: str) -> Dict[int, List[Tuple[int, int]]]:
    """Parses the Cranfield file with the expected doc similarity"""
    test_fd = open(test_path, 'r+')
    # Gets the list of the doc_ids with most similarity sorted
    sim_queries = {}
    test_re = re.compile('(\d+) (\d+) (-?\d+)')
    for line in test_fd.readlines():
        m = test_re.match(line)
        if m is not None:
            query_id = int(m.group(1))
            doc_id = int(m.group(2))
            relevance = int(m.group(3))
            try:
                sim_queries[query_id].append((doc_id, relevance))
            except KeyError:
                sim_queries[query_id] = [(doc_id, relevance)]
    for query in sim_queries.keys():
        sim_queries[query].sort(key=lambda x: x[1])
    return sim_queries


def test_model(queries: Dict[int, str], sim_queries: Dict[int, List[Tuple[int, int]]], model: MRI, evaluation):
    """
    Given the Cranfield queries and the similarity docs per query of the Cranfield test file
    and compares the performance of the ri model given a certain metric.
    """
    total_score = []
    for query_id, text in queries.items():
        # IDs of the relevant docs
        try:
            doc_ids_rel = [d[0] for d in sim_queries[query_id]]
        except KeyError:
            # the queries in the test and in the file doesn't have the same length
            break
        # Recovering the docs
        ranking = model.ranking_function(text)
        # IDs of the recovered docs
        doc_ids_rec = [d[0] for d in ranking[:100]]
        total_score.append(evaluation(doc_ids_rel, doc_ids_rec))
    return sum(total_score) / len(total_score)


def save_test_file(results):
    json.dump(results, open('../../resources/cluster/cran_rel.json', 'w+'))


def load_test_file():
    return json.load(open('../../resources/cluster/cran_rel.json', 'r+'))


def test(mri_model: MRI, evaluation_metric):
    queries = parse_query('../../resources/corpus/cran/cran.qry')
    # sim_queries = parse_test_file('../../resources/cran/cranqrel')
    # save_test_file(sim_queries)
    sim_queries = load_test_file()
    # score = test_model(queries, sim_queries, mri_model, evaluation_metric)
    # print(score)


if __name__ == '__main__':
    mri = VectorMRI(CranCorpusAnalyzer('../../resources/corpus/cran/cran.all.1400'))
    test(mri, f1_score)

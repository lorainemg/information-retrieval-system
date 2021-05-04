import re
from models import MRI, VectorMRI
from typing import List, Dict, Tuple
from corpus import CranCorpusAnalyzer

def parse_query(query_path: str) -> Dict[int, str]:
    """Parses the query cranfield file"""
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


def test_model(queries: Dict[int, str], sim_queries: Dict[int, List[Tuple[int, int]]], model: MRI):
    for query_id, text in queries.items():
        gold = sim_queries[query_id]
        result = model.ranking_function(text)


def test():
    queries = parse_query('../../resources/cran/cran.qry')
    sim_queries = parse_test_file('../../resources/cran/cranqrel')
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    mri = VectorMRI(analyzer)
    test_model(queries, sim_queries, mri)


if __name__ == '__main__':
    test()

from typing import Dict
from tests.lisa_tests import LisaTestManager
from pathlib import Path
import re

class NplTestManager(LisaTestManager):
    # npl test files are similar with lisa
    def __init__(self, query_path: Path, relation_path: Path, *, name='npl'):
        super().__init__(query_path, relation_path, name=name)

    def parse_test_file(self, relation_path: Path) -> Dict[int, Dict[int, int]]:
        test_fd = open(relation_path, 'r+')
        # Gets the list of the doc_ids with most similarity sorted
        sim_queries = {}
        test_re = re.compile('(\d+)')
        current_docs = []
        query_id = 0
        for line in test_fd.readlines():
            m = test_re.match(line)
            if m is not None:
                query_id = int(m.group(1))
            elif line.endswith('/\n'):
                for doc_id in current_docs:
                    relevance = 0
                    try:
                        sim_queries[query_id][doc_id] = relevance
                    except KeyError:
                        try:
                            sim_queries[query_id].update({doc_id: relevance})
                        except KeyError:
                            sim_queries[query_id] = {doc_id: relevance}
                current_docs = []
            else:
                current_docs += [int(word) for word in line.split()]
        return sim_queries




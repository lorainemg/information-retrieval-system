from tests.cranfield_tests import CranfieldTestManager
from pathlib import Path
from typing import Dict
import re


class CisiTestManager(CranfieldTestManager):
    def __init__(self, query_path: Path, relation_path: Path, *, name='cisi'):
        super().__init__(query_path, relation_path, name=name)

    def parse_test_file(self, relation_path: Path) -> Dict[int, Dict[int, int]]:
        """Parses the Cisi file with the expected doc similarity"""
        test_fd = open(relation_path, 'r+')
        # Gets the list of the doc_ids with most similarity sorted
        sim_queries = {}
        test_re = re.compile('\s+(\d+)\s+(\d+)')
        for line in test_fd.readlines():
            m = test_re.match(line)
            if m is not None:
                query_id = int(m.group(1))
                doc_id = int(m.group(2))
                relevance = 0
                try:
                    sim_queries[query_id][doc_id] = relevance
                except KeyError:
                    try:
                        sim_queries[query_id].update({doc_id: relevance})
                    except KeyError:
                        sim_queries[query_id] = {doc_id: relevance}
        return sim_queries

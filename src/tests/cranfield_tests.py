"""Tests a IR Model with a given metric with the Cranfield documents"""
import re
from pathlib import Path
from typing import Dict
from tests.test_manager import TestFileManager


class CranfieldTestManager(TestFileManager):
    def __init__(self, query_path: Path, relation_path: Path, *, name='cran'):
        super().__init__(query_path, relation_path, name=name)

    def parse_query(self, query_path: Path) -> Dict[int, str]:
        """Parses the query Cranfield file"""
        query_fd = open(query_path, 'r+')

        queries = {}
        current_id: int = None
        current_lines = []
        id_re = re.compile(r'\.I (\d+)')
        # marca cuando empieza el texto del documento actual
        getting_words = False
        n = 1
        for line in query_fd.readlines():
            m = id_re.match(line)
            # se empieza una nueva query
            if m is not None:
                # había un documento actual que se guarda en la lista de documentos
                if len(current_lines) > 0:
                    queries[current_id] = " ".join(current_lines)
                    n += 1
                current_id = n
                current_lines = []
                getting_words = False
            elif line.startswith('.W'):
                getting_words = True
            elif getting_words:
                current_lines.append(line[:-1])
        return queries

    def parse_test_file(self, relation_path: Path) -> Dict[int, Dict[int, int]]:
        """Parses the Cranfield file with the expected doc similarity"""
        test_fd = open(relation_path, 'r+')
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
                    sim_queries[query_id][doc_id] = relevance
                except KeyError:
                    try:
                        sim_queries[query_id].update({doc_id: relevance})
                    except KeyError:
                        sim_queries[query_id] = {doc_id: relevance}
        for query in sim_queries.keys():
            sim_queries[query] = dict(sorted(sim_queries[query].items(), key=lambda x: x[1]))
        return sim_queries

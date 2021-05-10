from typing import Dict
from tests.test_manager import TestFileManager
from pathlib import Path
import re


class LisaTestManager(TestFileManager):
    def __init__(self, query_path: Path, relation_path: Path, *, name='lisa'):
        super().__init__(query_path, relation_path, name=name)

    def parse_query(self, query_path: Path) -> Dict[int, str]:
        query_fd = open(query_path, 'r+')

        queries = {}
        current_id: int = None
        current_lines = []
        id_re = re.compile(r'(\d+)')
        for line in query_fd.readlines():
            m = id_re.match(line)
            # se empieza una nueva query
            if m is not None:
                # había un documento actual que se guarda en la lista de documentos
                if len(current_lines) > 0:
                    queries[current_id] = " ".join(current_lines)
                current_id = int(m.group(1))
                current_lines = []
            else:
                current_lines.append(line[:-1])
        return queries

    def parse_test_file(self, relation_path: Path) -> Dict[int, Dict[int, int]]:
        test_fd = open(relation_path, 'r+')
        # Gets the list of the doc_ids with most similarity sorted
        sim_queries = {}
        test_re = re.compile(r'Query (\d+)')
        rel_re = re.compile(r'\d+ Relevant Refs:')
        query_id: int = None
        next_line_is_doc = False
        for line in test_fd.readlines():
            m = test_re.match(line)
            if m is not None:
                query_id = int(m.group(1))
            elif rel_re.match(line) is not None:
                next_line_is_doc = True
            elif next_line_is_doc:
                doc_ids = [int(word) for word in line.split()[:-1]]
                for doc_id in doc_ids:
                    relevance = 0
                    try:
                        sim_queries[query_id][doc_id] = relevance
                    except KeyError:
                        try:
                            sim_queries[query_id].update({doc_id: relevance})
                        except KeyError:
                            sim_queries[query_id] = {doc_id: relevance}
                next_line_is_doc = False
        return sim_queries




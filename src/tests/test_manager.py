from models import MRI
from pathlib import Path
from typing import Dict
from query import QueryParser
import json


class TestFileManager:
    def __init__(self, query_path: Path, relation_path: Path, *, name='test'):
        """
        Initializes the test manager
        :param query_path: path where the expected query path file is located
        :param relation_path: path where is located the expected relation standard file
        """
        self.name = name
        try:
            self.load_test_file()
        except FileNotFoundError:
            self.queries = self.parse_query(query_path)
            self.similarity_queries = self.parse_test_file(relation_path)
            self.save_test_files()
        self.query_parser = QueryParser()

    def parse_query(self, query_path: Path) -> Dict[int, str]:
        """Parses the file located in `query_path`"""
        raise NotImplementedError

    def parse_test_file(self, relation_path: Path) -> Dict[int, Dict[int, int]]:
        """Parses the file located in `relation_path`"""
        raise NotImplementedError

    def test_model(self, model: MRI, evaluation):
        """
        Tests the model with the queries in `query_path` according to gold standard
        in `relation_path` using the given model and the given evaluation
        :param model: an instance of a mri model
        :param evaluation: on of the evaluation metrics included in `evaluation.py`
        """
        total_score = []
        for query_id, text in self.queries.items():
            # IDs of the relevant docs
            try:
                doc_ids_rel = [int(doc_id) for doc_id in self.similarity_queries[query_id].keys()]
            except KeyError:
                # the queries in the test and in the file doesn't have the same length
                break
            # Recovering the docs
            ranking = model.ranking_function(self.query_parser(text, model.doc_analyzer.index))
            # IDs of the recovered docs
            doc_ids_rec = [d[0] for d in ranking[:len(doc_ids_rel)]]
            total_score.append(evaluation(doc_ids_rel, doc_ids_rec))
        return sum(total_score) / len(total_score)

    def save_test_files(self):
        json.dump(self.queries, open(f'../resources/test/{self.name}_queries.json', 'w+'))
        json.dump(self.similarity_queries, open(f'../resources/test/{self.name}_sim_queries.json', 'w+'))

    def load_test_file(self):
        self.queries = json.load(open(f'../resources/test/{self.name}_queries.json', 'r+'))
        self.similarity_queries = json.load(open(f'../resources/test/{self.name}_sim_queries.json', 'r+'))


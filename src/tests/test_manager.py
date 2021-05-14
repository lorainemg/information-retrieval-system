from system import IRSystem
from pathlib import Path
from typing import Dict, List
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

    def test_model(self, system: IRSystem, evaluations):
        """
        Tests the model with the queries in `query_path` according to gold standard
        in `relation_path` using the given model and the given evaluation
        :param model: an instance of a mri model
        :param evaluation: on of the evaluation metrics included in `evaluation.py`
        """
        total_score = []
        n = 0
        for query_id, text in self.queries.items():
            # IDs of the relevant docs
            try:
                doc_ids_rel = [int(doc_id) for doc_id in self.similarity_queries[query_id].keys()]
            except KeyError:
                # the queries in the test and in the file doesn't have the same length
                continue
            # Recovering the docs
            documents = system.make_query(text)
            # IDs of the recovered docs
            doc_ids_rec = [doc.id for doc in documents[:len(doc_ids_rel)]]
            results = {}
            for evaluation in evaluations:
                results[evaluation.__name__] = evaluation(doc_ids_rel, doc_ids_rec)
            total_score.append(results)
            n += 1
        print(n)
        return self.get_final_results(total_score)

    def get_final_results(self, results: List[Dict[str, float]]):
        final_results = {}
        for scores in results:
            for name, score in scores.items():
                try:
                    final_results[name] += score
                except KeyError:
                    final_results[name] = score
        for name, score in final_results.items():
            final_results[name] = score / len(results)
        return final_results

    def save_test_files(self):
        json.dump(self.queries, open(f'../resources/test/{self.name}_queries.json', 'w+'))
        json.dump(self.similarity_queries, open(f'../resources/test/{self.name}_sim_queries.json', 'w+'))

    def load_test_file(self):
        self.queries = json.load(open(f'../resources/test/{self.name}_queries.json', 'r+'))
        self.similarity_queries = json.load(open(f'../resources/test/{self.name}_sim_queries.json', 'r+'))


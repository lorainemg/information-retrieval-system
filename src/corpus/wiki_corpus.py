from corpus.corpus import CorpusAnalyzer
from pathlib import Path
from tools import Document
import json


class WikiCorpusAnalyzer(CorpusAnalyzer):
    def __init__(self, corpus_path: Path, *, name='wiki'):
        super().__init__(corpus_path, name=name)

    def parse_documents(self, corpus_path: Path):
        docs = json.load(open(corpus_path, 'r', encoding='utf-8'))
        for id, (doc_tile, doc_text) in enumerate(docs.items()):
            text = self.preprocess_text(doc_text)
            summary = doc_text[:500] + '...'
            self.documents.append(Document(id, text, doc_tile, summary))
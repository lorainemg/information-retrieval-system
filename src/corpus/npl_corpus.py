from corpus.corpus import CorpusAnalyzer
from tools import Document
from typing import List
from pathlib import Path
import re


class NplCorpusAnalyzer(CorpusAnalyzer):
    """Corpus analyzer for the npl dataset"""
    def __init__(self, corpus_path: Path, *, name='npl'):
        super().__init__(corpus_path, name=name)

    def parse_documents(self, corpus_path: Path):
        corpus_fd = open(corpus_path, 'r+')
        current_id: int = None
        corpus_re = re.compile(r'(\d+)')

        current_lines: List[str] = []
        for line in corpus_fd.readlines():
            match = corpus_re.match(line)
            if match is not None:
                if len(current_lines) > 0:
                    tokens = self.preprocess_text(" ".join(current_lines), stemming=False)
                    summary = " ".join(current_lines[:20] + ['...'])
                    self.documents.append(Document(current_id, tokens, "", summary))
                current_id = int(match.group(1))
                current_lines = []
            elif not line.endswith('/\n'):
                current_lines.append(line[:-1])

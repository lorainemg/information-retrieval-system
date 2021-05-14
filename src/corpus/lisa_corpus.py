from corpus.corpus import CorpusAnalyzer
from tools import Document
from typing import List
from pathlib import Path
from os import listdir
import re


class LisaCorpusAnalyzer(CorpusAnalyzer):
    """Corpus analyzer for the lisa dataset"""
    def __init__(self, corpus_path: Path, *, name='lisa', stemming=False):
        self.id_re = re.compile('Document\s+(\d+)')
        CorpusAnalyzer.__init__(self, corpus_path, name=name, stemming=stemming)

    def parse_documents(self, corpus_path: Path):
        corpus_re = re.compile(r'LISA[012345].(\d+)')
        for file in listdir(corpus_path):
            if corpus_re.match(file):
                self.parse_doc(corpus_path / file)

    def parse_doc(self, file_path: Path):
        corpus_fd = open(file_path)
        current_id: int = None

        current_lines: List[str] = []
        getting_words = False

        current_title: list = []
        getting_title = False
        for line in corpus_fd.readlines():
            match = self.id_re.match(line)
            if match is not None:
                if len(current_lines) > 0:
                    tokens = self.preprocess_text(" ".join(current_lines))
                    title = self.title_preprocessing(current_title)
                    summary = " ".join(current_lines[:20] + ['...'])
                    self.documents.append(Document(current_id, tokens, title, summary))
                current_id = int(match.group(1))
                current_title = []
                current_lines = []
                getting_title = True
            elif line.startswith('*'):
                getting_words = False
            elif line.isspace():
                getting_title = False
                getting_words = True
            elif getting_title:
                current_title.append(line[:-1])
            elif getting_words:
                current_lines.append(line[:-1])

    def title_preprocessing(self, title: List[str]):
        return ' '.join(title).capitalize()

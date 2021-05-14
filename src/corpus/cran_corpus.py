from corpus import CorpusAnalyzer
from typing import Pattern, List
import re
from tools import Document


class CranCorpusAnalyzer(CorpusAnalyzer):
    """
    The cran corpus has the following structure:
        .I [#] (id of the document)
        .T (title) (can occupy more than one line)
        .A (author) (can occupy more than one line)
        .B (something about the origin)
        .W (the words of the document) (usually they occupy more than one line)
        the first line is the title
    """

    def __init__(self, corpus_path, *, name='cran', stemming=False):
        # Regular expresion to extract the id of the document
        self.id_re: Pattern = re.compile(r'\.I (\d+)')
        CorpusAnalyzer.__init__(self, corpus_path, name=name, stemming=stemming)

    def parse_documents(self, corpus_path):
        corpus_fd = open(corpus_path, 'r')
        # current document that is being built
        current_id: int = None
        current_lines: List[str] = []
        # marca cuando empieza el texto del documento actual
        getting_words = False
        current_title: list = []
        getting_title = False
        for i, line in enumerate(corpus_fd.readlines()):
            m = self.id_re.match(line)
            # se empieza un nuevo documento
            if m is not None:
                # había un documento actual que se guarda en la lista de documentos
                if len(current_lines) > 0:
                    tokens = self.preprocess_text(" ".join(current_lines))
                    title = self.title_preprocessing(current_title)
                    summary = " ".join(current_lines[:20] + ['...'])
                    self.documents.append(Document(current_id, tokens, title, summary))
                current_id = int(m.group(1))
                current_lines = []
                getting_words = False
                current_title = []
            elif line.startswith('.T'):
                getting_title = True
            elif line.startswith('.W'):
                getting_words = True
            elif line.startswith('.A'):
                getting_title = False
            elif line.startswith('.X'):
                getting_words = False
            elif getting_words:
                current_lines.append(line[:-1])
            elif getting_title:
                current_title.append(line[:-1])

    def title_preprocessing(self, title: List[str]):
        title[0] = title[0].capitalize()
        if title[-1][-1] == '.':
            title[-1] = title[-1][:-1]
        return " ".join(title)

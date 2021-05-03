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
        .B (something about where was published)
        .W (the words of the document) (usually they occupy more than one line)
        the first line is the title
    """

    def __init__(self, corpus_fd):
        CorpusAnalyzer.__init__(self, corpus_fd)
        # Regular expresion to extract the id of the document
        self.id_re: Pattern = re.compile(r'\.I (\d+)')

    def parse_documents(self):
        lines = self.corpus_fd.readlines()
        # current document that is being built
        current_id: int = None
        current_lines: List[str] = None
        # marca cuando empieza el texto del documento actual
        getting_words = False
        for line in lines:
            m = self.id_re.match(line)
            # se empieza un nuevo documento
            if m is not None:
                # había un documento actual que se guarda en la lista de documentos
                if current_lines is not None:
                    # probablemente haga el preprocesamiento aquí
                    tokens = self.tokenize(" ".join(current_lines))
                    self.documents[current_id] = Document(current_id, tokens)
                current_id = m.group(1)
                current_lines = []
            elif line.startswith('.W'):
                getting_words = True
            elif getting_words:
                current_lines.append(line)

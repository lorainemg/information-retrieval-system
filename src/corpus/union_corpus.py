from pathlib import Path
from os import listdir

from corpus.corpus import CorpusAnalyzer
from corpus.cisi_corpus import CisiCorpusAnalyzer
from corpus.cran_corpus import CranCorpusAnalyzer
from corpus.lisa_corpus import LisaCorpusAnalyzer
from corpus.npl_corpus import NplCorpusAnalyzer
from tools import Document


class UnionCorpusAnalyzer(CorpusAnalyzer):
    def __init__(self, corpus_path: Path, *, name='union'):
        super().__init__(corpus_path, name=name)

    """Corpus to contain the union of all documents"""
    def parse_documents(self, corpus_path: Path):
        """
        Fills the document list parsing the file
        :param corpus_path: Folder containing 4 folder of corpus: cisi, cran, lisa, npl
        """
        for file in listdir(corpus_path):
            if file == 'cisi':
                cisi_analyzer = CisiCorpusAnalyzer(corpus_path / file / 'CISI.ALL')
            elif file == 'cran':
                cran_analyzer = CranCorpusAnalyzer(corpus_path / file / 'cran.all.1400')
            elif file == 'lisa':
                lisa_analyzer = LisaCorpusAnalyzer(corpus_path / file)
            elif file == 'npl':
                npl_analyzer = NplCorpusAnalyzer(corpus_path / file / 'doc-text')
        for i, doc in enumerate(cisi_analyzer.documents + cran_analyzer.documents + lisa_analyzer.documents + npl_analyzer.documents):
            self.documents.append(Document(i, doc.tokens, doc.title, doc.summary))

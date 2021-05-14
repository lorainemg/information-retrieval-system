from corpus.cran_corpus import CranCorpusAnalyzer


class CisiCorpusAnalyzer(CranCorpusAnalyzer):
    """Cisi corpus has a similar structure of the Cran Corpus"""
    def __init__(self, corpus_path, *, name='cisi', stemming=False):
        super().__init__(corpus_path, name=name, stemming=stemming)

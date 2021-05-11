"""File to test the different models and corpus"""
import corpus
import tests
from models import VectorMRI
from pathlib import Path
import evaluation


if __name__ == '__main__':
    # analyzer = corpus.CranCorpusAnalyzer(Path('../ratings/corpus/cran/cran.all.1400'))
    # test_manager = tests.CranfieldTestManager(Path('../ratings/corpus/cran/cran.qry'),
    #                                           Path('../ratings/corpus/cran/cranqrel'))
    # analyzer = corpus.CisiCorpusAnalyzer(Path('../ratings/corpus/cisi/CISI.ALL'))
    # test_manager = tests.CisiTestManager(Path('../ratings/corpus/cisi/CISI.QRY'),
    #                                      Path('../ratings/corpus/cisi/CISI.REL'))
    # analyzer = corpus.LisaCorpusAnalyzer(Path('../ratings/corpus/lisa'))
    # test_manager = tests.LisaTestManager(Path('../ratings/corpus/lisa/LISA.QUE'),
    #                                      Path('../ratings/corpus/lisa/LISA.REL'))
    analyzer = corpus.NplCorpusAnalyzer(Path('../resources/corpus/npl/doc-text'))
    test_manager = tests.NplTestManager(Path('../resources/corpus/npl/query-text'),
                                        Path('../resources/corpus/npl/rlv-ass'))

    score = test_manager.test_model(VectorMRI(analyzer), evaluation.f1_score)
    print(score)
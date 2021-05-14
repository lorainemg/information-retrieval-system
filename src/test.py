"""File to test the different models and corpus"""
import corpus
import tests
from models import VectorMRI
from system import IRSystem
from pathlib import Path
import evaluation
import json
from pprint import pprint


def scores_depending_on_stemming(evaluations, cran_test_manager, cisi_test_manager, lisa_test_manager, npl_test_manager,
                                 stemming):
    cran_analyzer = corpus.CranCorpusAnalyzer(Path('../resources/corpus/cran/cran.all.1400'), stemming=stemming)
    cran_score = cran_test_manager.test_model(IRSystem(VectorMRI(cran_analyzer)), evaluations)

    cisi_analyzer = corpus.CisiCorpusAnalyzer(Path('../resources/corpus/cisi/CISI.ALL'), stemming=stemming)
    cisi_score = cisi_test_manager.test_model(IRSystem(VectorMRI(cisi_analyzer)), evaluations)

    lisa_analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'), stemming=stemming)
    lisa_score = lisa_test_manager.test_model(IRSystem(VectorMRI(lisa_analyzer)), evaluations)

    npl_analyzer = corpus.NplCorpusAnalyzer(Path('../resources/corpus/npl/doc-text'), stemming=stemming)
    npl_score = npl_test_manager.test_model(IRSystem(VectorMRI(npl_analyzer)), evaluations)
    return {
        'cran': cran_score,
        'cisi': cisi_score,
        'lisa': lisa_score,
        'npl': npl_score
    }


if __name__ == '__main__':
    evaluations = [evaluation.precision_score, evaluation.recall_score, evaluation.f1_score]

    cran_test_manager = tests.CranfieldTestManager(Path('../resources/corpus/cran/cran.qry'),
                                                   Path('../resources/corpus/cran/cranqrel'))
    cisi_test_manager = tests.CisiTestManager(Path('../resources/corpus/cisi/CISI.QRY'),
                                              Path('../resources/corpus/cisi/CISI.REL'))
    lisa_test_manager = tests.LisaTestManager(Path('../resources/corpus/lisa/LISA.QUE'),
                                              Path('../resources/corpus/lisa/LISA.REL'))
    npl_test_manager = tests.NplTestManager(Path('../resources/corpus/npl/query-text'),
                                            Path('../resources/corpus/npl/rlv-ass'))

    scores_no_stemming = scores_depending_on_stemming(evaluations, cran_test_manager, cisi_test_manager, \
                                                      lisa_test_manager, npl_test_manager, False)
    print('Score with no stemming')
    pprint(scores_no_stemming)

    scores_stemming = scores_depending_on_stemming(evaluations, cran_test_manager, cisi_test_manager, \
                                                   lisa_test_manager, npl_test_manager, True)
    print('Score with stemming')
    pprint(scores_stemming)

    json.dump({
        'scores_no_stemming': scores_no_stemming,
        'scores_stemming': scores_stemming
    }, open(Path('../resources/results.json'), 'w'))


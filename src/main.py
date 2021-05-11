from system import IRSystem
import corpus
from models import VectorMRI
from query_expansion import query_expansion
from query import QueryParser
from pathlib import Path
from clustering import ClusterManager
from document_recommendation import DocumentRecommender

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lisa_analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'))
    lisa_system = IRSystem(VectorMRI(lisa_analyzer))

    #
    # cran_analyzer = corpus.CranCorpusAnalyzer(Path('../ratings/corpus/cran/cran.all.1400'))
    # cran_system = IRSystem(VectorMRI(cran_analyzer))
    #
    # cisi_analyzer = corpus.CisiCorpusAnalyzer(Path('../ratings/corpus/cisi/CISI.ALL'))
    # cisi_system = IRSystem(VectorMRI(cisi_analyzer))
    #
    # npl_analyzer = corpus.NplCorpusAnalyzer(Path('../ratings/corpus/npl/doc-text'))
    # npl_system = IRSystem(VectorMRI(npl_analyzer))
    #
    # all_analyzer = corpus.UnionCorpusAnalyzer(Path('../ratings/corpus'))
    # all_system = IRSystem(VectorMRI(all_analyzer))

    system = lisa_system
    mri = system.model
    analyzer = system.corpus
    #
    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    docs = system.make_query(query)
    similarity = mri.ranking_function(QueryParser()(query, analyzer.index))

    relevance = [doc_id for doc_id, freq in similarity[:100] if freq > 10 and freq % 2 == 0]
    total_docs = [doc_id for doc_id, _ in similarity[:100]]
    new_query = system.user_feedback(query, relevance, total_docs)
    #
    # q_vect = query_expansion(QueryParser().parse(query), lisa_analyzer)
    # print(q_vect)

    docs = lisa_system.get_recommended_documents()
    print(docs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

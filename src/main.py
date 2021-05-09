from system import IRSystem
import corpus
from models import VectorMRI
from query_expansion import query_expansion_with_nltk
from query import QueryParser
from pathlib import Path

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'))
    # analyzer = corpus.NplCorpusAnalyzer(Path('../resources/corpus/npl/doc-text'))
    # analyzer = corpus.CranCorpusAnalyzer(Path('../resources/corpus/cran/cran.all.1400'))
    # analyzer = corpus.CisiCorpusAnalyzer(Path('../resources/corpus/cisi/CISI.ALL'))
    # analyzer = corpus.UnionCorpusAnalyzer(Path('../resources/corpus'))
    mri = VectorMRI(analyzer)
    system = IRSystem(mri, analyzer)


    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    docs = system.make_query(query)
    similarity = mri.ranking_function(QueryParser()(query, analyzer.index))

    relevance = [doc_id for doc_id, freq in similarity[:100] if freq > 10 and freq % 2 == 0]
    total_docs = [doc_id for doc_id, _ in similarity[:100]]
    new_query = system.user_feedback(query, relevance, total_docs)

    q_vect = query_expansion_with_nltk(QueryParser().parse(query))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

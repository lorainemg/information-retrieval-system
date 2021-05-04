from system import IRSystem
from corpus import CranCorpusAnalyzer
from models import VectorMRI
from query_expansion import query_expansion_with_nltk
from query import QueryParser

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    mri = VectorMRI(analyzer)
    system = IRSystem(mri, analyzer)

    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    docs = system.make_query(query)
    similarity = mri.ranking_function(query)

    relevance = [doc_id for doc_id, freq in similarity[:100] if freq > 10 and freq % 2 == 0]
    total_docs = [doc_id for doc_id, _ in similarity[:100]]
    new_query = system.user_feedback(query, relevance, total_docs)

    q_vect = query_expansion_with_nltk(QueryParser().parse(query))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

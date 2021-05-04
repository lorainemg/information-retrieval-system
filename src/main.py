from corpus import CranCorpusAnalyzer
from models import VectorMRI
from retroalimentation import RocchioAlgorithm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    mri = VectorMRI(analyzer)
    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    similarity = mri.ranking_function(query)
    print(similarity)
    print(mri.get_similarity_docs(query))

    n = similarity[0][1]
    relevance = [(ti, freq/n) for ti, freq in similarity[:100]]
    rocchio = RocchioAlgorithm(query, analyzer, relevance)
    nquery_vect = rocchio()
    print(nquery_vect)
    ntokens = rocchio.get_tokens_by_vector(nquery_vect)
    print(ntokens)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

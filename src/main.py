from corpus import CranCorpusAnalyzer
from models import VectorMRI

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    mri = VectorMRI(analyzer)
    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    similarity = mri.ranking_function(query)
    print(similarity)
    print(mri.get_similarity_docs(query))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

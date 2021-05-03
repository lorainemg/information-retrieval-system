from corpus import CranCorpusAnalyzer
from pprint import pprint

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    analyzer.parse_documents()
    analyzer.create_document_index()
    pprint(analyzer.documents)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

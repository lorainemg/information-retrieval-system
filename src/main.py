from system import IRSystem
import corpus
from models import VectorMRI
from query_expansion import query_expansion
from query import QueryParser
from pathlib import Path
from pprint import pprint


def all_steps(query: str, system: IRSystem):
    analyzer = system.corpus
    print(f'Query for the {analyzer.name} corpus')
    docs, similarity = system.make_query(query, ranking=True)
    print('First 20 results for the query:')
    for doc in docs[:20]:
        print(doc)
    print("Doing pseudo-feedback")
    system.pseudo_feedback(query, similarity)
    print('Doing query-expansion:')
    new_expansions = query_expansion(QueryParser().parse(query), analyzer)
    print(new_expansions)
    print('Getting recommended documents:')
    docs = system.get_recommended_documents()
    for doc in docs:
        if doc is not None:
            print(doc)
    print()


def get_analyzer(type_):
    if type_ == 'lisa':
        analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'))
        return IRSystem(VectorMRI(analyzer))
    elif type_ == 'npl':
        analyzer = corpus.NplCorpusAnalyzer(
            Path('../resources/corpus/npl/doc-text'))
        return IRSystem(VectorMRI(analyzer))
    elif type_ == 'cran':
        analyzer = corpus.CranCorpusAnalyzer(
            Path('../resources/corpus/cran/cran.all.1400'))
        return IRSystem(VectorMRI(analyzer))
    elif type_ == 'cisi':
        analyzer = corpus.CisiCorpusAnalyzer(
            Path('../resources/corpus/cisi/CISI.ALL'))
    elif type_ == 'wiki':
        analyzer = corpus.WikiCorpusAnalyzer(
            Path('../resources/corpus/wiki_docs.json'))
    elif type_ == 'all':
        analyzer = corpus.UnionCorpusAnalyzer(Path('../resources/corpus'))
    else:
        return None
    return IRSystem(VectorMRI(analyzer))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        type_ = input('Please choose a corpus [cran, cisi, lisa, npl, wiki, all]:\n')
        system = get_analyzer(type_)
        if system is None:
            print('Please choose a valid corpus')
            continue
        else:
            break
    while True:
        query = input('Make a query:\n')
        all_steps(query, system)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

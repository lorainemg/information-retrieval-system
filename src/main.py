from system import IRSystem
import corpus
from models import VectorMRI
from query_expansion import query_expansion
from query import QueryParser
from pathlib import Path
from crawler import WikiCrawler


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # wiki_analyzer = corpus.WikiCorpusAnalyzer(Path('../resources/corpus/wiki_docs.json'))

    # lisa_analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'))
    # lisa_system = IRSystem(VectorMRI(wiki_analyzer))

    #
    cran_analyzer = corpus.CranCorpusAnalyzer(Path('../ratings/corpus/cran/cran.all.1400'))
    cran_system = IRSystem(VectorMRI(cran_analyzer))

    # cisi_analyzer = corpus.CisiCorpusAnalyzer(Path('../ratings/corpus/cisi/CISI.ALL'))
    # cisi_system = IRSystem(VectorMRI(cisi_analyzer))
    #
    # npl_analyzer = corpus.NplCorpusAnalyzer(Path('../ratings/corpus/npl/doc-text'))
    # npl_system = IRSystem(VectorMRI(npl_analyzer))
    #
    # all_analyzer = corpus.UnionCorpusAnalyzer(Path('../ratings/corpus'))
    # all_system = IRSystem(VectorMRI(all_analyzer))

    # system = lisa_system
    # mri = system.model
    # analyzer = system.corpus
    # # #
    query = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .'
    # docs = system.make_query(query)
    # print(docs)
    # similarity = mri.ranking_function(QueryParser()(query, analyzer.index))
    #
    # relevance = [doc_id for doc_id, freq in similarity[:100] if freq > 10 and freq % 2 == 0]
    # total_docs = [doc_id for doc_id, _ in similarity[:100]]
    # new_query = system.user_feedback(query, relevance, total_docs)
    # #
    new_expantions = query_expansion(QueryParser().parse(query), cran_analyzer)
    print(new_expantions)
    #
    # docs = lisa_system.get_recommended_documents()
    # print(docs)

    # crawler = WikiCrawler()
    # crawler.crawl(10)
    # page = crawler.go_to_link('file:///media/loly/02485E43485E359F/_Escuela/__UH/5to/SI/Proyecto final/Info/test_pages/International School of Dongguan - Wikipedia.html')
    # crawler.get_document_info(page)
    # crawler.get_links(page)
    # print(crawler.documents)
    # print(crawler.links)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import streamlit as st
import numpy as np
from pathlib import Path
import time

from streamlit.caching import cache
import visual.session_state

from typing import List, Dict

import corpus
from models import VectorMRI
from system import IRSystem
from tools import Document

# Logic


def create_system(type: str) -> IRSystem:
    print('here')
    if type == 'lisa':
        analyzer = corpus.LisaCorpusAnalyzer(Path('../resources/corpus/lisa'))
        return IRSystem(VectorMRI(analyzer))

    if type == 'npl':
        analyzer = corpus.NplCorpusAnalyzer(
            Path('../resources/corpus/npl/doc-text'))
        return IRSystem(VectorMRI(analyzer))

    if type == 'cran':
        analyzer = corpus.CranCorpusAnalyzer(
            Path('../resources/corpus/cran/cran.all.1400'))
        return IRSystem(VectorMRI(analyzer))

    if type == 'cisi':
        analyzer = corpus.CisiCorpusAnalyzer(
            Path('../resources/corpus/cisi/CISI.ALL'))
        return IRSystem(VectorMRI(analyzer))
    if type == 'all':
        analyzer = corpus.UnionCorpusAnalyzer(Path('../resources/corpus'))
        return IRSystem(VectorMRI(analyzer))


@cache(show_spinner=False, suppress_st_warning=True, persist=True)
def get_query_result(query: str) -> List[Document]:
    return state.system.make_query(query)

# UI


def create_item(title, description=None,  on_feedback=None):

    contentCol, indxCol = st.beta_columns([11, 1])
    st.text("")
    # indxCol.subheader(f'{index}.')
    indxCol.text("")
    contentCol.subheader(f'**{title}**')
    contentCol.write(description)

    if indxCol.checkbox(label="⭐", key=title):
        on_feedback(True)
    else:
        on_feedback(False)


page_size = st.sidebar.number_input(
    label='Number of results', value=20, step=10, help='The number of result show for every search')
state = visual.session_state.get(system=None, feedback={}, query=None)

corpus_type = st.sidebar.selectbox(label='Select Corpus', options=['all', "cisi", "cran", "lisa", "npl"], index= 0)
state.system = create_system(corpus_type)

system: IRSystem = state.system
feedback: Dict[str, List[int]] = state.feedback

# Search Bar`
query = st.text_input(label='Search something', key='query')
feedback[query] = []


if(query is None or query == ""):
    st.stop()

documents: List[Document] = get_query_result(query)
print(documents)
st.success(f"We found {len(documents)} matches")


def set_feedback(query: str, id: str):
    def _set(value: bool):
        if value:
            feedback[query].append(id)
        else:
            try:
                feedback[query].remove(id)
            except:
                pass
    return _set


with st.beta_container():
    for doc in documents[:page_size]:
        create_item(doc.title, doc.summary,
                    on_feedback=set_feedback(query, doc.id)
                    )


if st.button(label="Send Feedback"):
    totals = [doc.id for doc in documents]
    relevants = feedback[query]
    system.user_feedback(query, relevants, totals)

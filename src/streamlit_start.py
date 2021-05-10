import streamlit as st
import numpy as np
import time

from streamlit.caching import cache
import visual.session_state

from typing import List, Dict

from corpus import CranCorpusAnalyzer
from models import VectorMRI
from system import IRSystem
from tools import Document

# Logic


def create_system() -> IRSystem:
    print("Creating System")
    analyzer = CranCorpusAnalyzer('../resources/cran/cran.all.1400')
    mri = VectorMRI(analyzer)
    return IRSystem(mri)

@cache(show_spinner= False, suppress_st_warning= True, persist= True)
def get_query_result(query: str) -> List[Document]:
    return state.system.make_query(query)

# UI


def create_item(title, description=None,  on_feedback=None):
    with st.beta_container():
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


state = visual.session_state.get(system=None, feedback={}, query=None)
if state.system == None:
    state.system = create_system()

print('System created')
system: IRSystem = state.system
feedback: Dict[str, List[int]] = state.feedback

# Search Bar
query = st.text_input(label='Search something', key='query')
feedback[query] = []
page_size = st.sidebar.number_input(
    label='Number of results', value=20, step=10, help='The number of result show for every search')


if(query is None or query == ""):
    st.stop()

documents: List[Document] = get_query_result(query)

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


for doc in documents[:page_size]:
    create_item(doc.title,
                f"""
    """,
                on_feedback=set_feedback(query, doc.id)
                )


if st.button(label="Send Feedback"):
    totals = [doc.id for doc in documents]
    relevants = feedback[query]
    system.user_feedback(query, relevants,totals )

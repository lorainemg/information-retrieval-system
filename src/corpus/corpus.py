"""Corpus to read and parse corpus"""
from typing import List, Tuple, Dict
from gensim.corpora import Dictionary
from pathlib import Path
import nltk
import pickle

from tools import Document
from utils import remove_punctuation, convert_to_lower, tokenize


class CorpusAnalyzer:
    def __init__(self, corpus_path: Path, *, name='corpus'):
        # A dictionary of documents where the keys are their identifier
        # and the value its a Document instance
        self.documents: List[Document] = []
        self.stemmer = nltk.PorterStemmer()
        self.index: Dictionary = None
        self.name = name
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.parse_documents(corpus_path)
        try:
            self.load_indexed_document()
        except FileNotFoundError or FileExistsError:
            self.create_document_index()
            self.vectors = self.docs2bows()
            self.save_indexed_document()

    def parse_documents(self, corpus_path: Path):
        """
        Parse the documents in the corpus_path and fills the documents instance,
        tokenizing and preprocessing them
        """
        raise NotImplementedError

    def preprocess_text(self, text, stemming=True):
        """Preprocess the text and returns a list of cleaned tokens"""
        text = remove_punctuation(text)
        text = convert_to_lower(text)
        tokens = tokenize(text)
        tokens = [tok for tok in tokens if tok not in self.stopwords]
        if stemming:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]):
        return [self.stemmer.stem(token) for token in tokens]

    def create_document_index(self, name='docs_index'):
        """This method creates a dictionary based on the taxonomy of keywords for each document."""
        docs = [d.tokens for d in self.documents]
        self.index = Dictionary(docs)

    def save_indexed_document(self):
        self.index.save(f'../resources/indexed_corpus/{self.name}/index.idx')
        # Save the corpus in the Matrix Market format
        pickle.dump(self.vectors, open(f'../resources/indexed_corpus/{self.name}/docs_vect.pkl', 'wb'))
        # MmCorpus.serialize('../resources/indexed_docs.mm', self.vectors)

    def load_indexed_document(self):
        self.index = Dictionary.load(f'../resources/indexed_corpus/{self.name}/index.idx')
        self.vectors = pickle.load(open(f'../resources/indexed_corpus/{self.name}/docs_vect.pkl', 'rb'))
        # self.vectors = MmCorpus.load('../resources/indexed_docs.mm')

    def id2doc(self, doc_id: int) -> Document:
        """Given a document id returns the documents that matches that id"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def docs2bows(self) -> List[Dict[int, int]]:
        """
        Converts all the document (the list of words) into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        return [dict(self.index.doc2bow(doc.tokens)) for doc in self.documents]

    def doc2bow(self, id: int) -> Dict[int, int]:
        """
        Converts the document matching the id into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        return self.vectors[id]

    def token2id(self, token: str):
        return self.index.token2id[token]

    def get_frequency(self, tok_id: int, doc_id: int) -> int:
        """Gets the frequency of a token in certain document"""
        vector = self.doc2bow(doc_id)
        try:
            return vector[tok_id]
        except KeyError:
            return 0

    def get_max_frequency(self, doc_id: int) -> Tuple[str, int]:
        """Gets the term of the max frequency in a certain document"""
        vector = self.doc2bow(doc_id)
        max_freq_id = max(vector.items(), key=lambda x: x[1])
        return self.index[max_freq_id[0]], max_freq_id[1]

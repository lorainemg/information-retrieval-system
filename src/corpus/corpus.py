"""Corpus to read and parse corpus"""
from typing import Dict, List, Tuple
from gensim.corpora import Dictionary
import nltk

from tools import Document
from utils import remove_punctuation, convert_to_lower, tokenize


class CorpusAnalyzer:
    def __init__(self, corpus_path: str):
        self.corpus_fd = open(corpus_path, 'r')
        # A dictionary of documents where the keys are their identifier
        # and the value its a Document instance
        self.documents: List[Document] = []
        self.stemmer = nltk.PorterStemmer()
        self.index: Dictionary = None
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.parse_documents()
        self.create_document_index()

    def parse_documents(self):
        """Parse the documents and fills the documents instance, tokenizing"""
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

    def id2doc(self, doc_id: int) -> Document:
        """Given a document id returns the documents that matches that id"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def docs2bows(self):
        """
        Converts all the document (the list of words) into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        vectors = [self.index.doc2bow(doc.tokens) for doc in self.documents]
        # corpora.MmCorpus.serialize('vsm_docs.mm', vectors)  # Save the corpus in the Matrix Market format
        return vectors

    def doc2bow(self, id):
        """
        Converts the document matching the id into the bag-of-words representation
        format = list of (token_id, token_count) 2-tuples.
        """
        try:
            return self.index.doc2bow(self.documents[id].tokens)
        except KeyError:
            return None

    def token2id(self, token: str):
        return self.index.token2id[token]

    def get_frequency(self, tok_id: int, doc_id: int) -> int:
        """Gets the frequency of a token in certain document"""
        vector = dict(self.doc2bow(doc_id))
        try:
            return vector[tok_id]
        except KeyError:
            return 0

    def get_max_frequency(self, doc_id: int) -> Tuple[str, int]:
        """Gets the term of the max frequency in a certain document"""
        vector = self.doc2bow(doc_id)
        max_freq_id = max(vector, key=lambda x: x[1])
        return self.index[max_freq_id[0]], max_freq_id[1]

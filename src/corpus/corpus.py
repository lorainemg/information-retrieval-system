"""Corpus to read and parse corpus"""
from typing import Dict, List
from gensim import corpora
import nltk

from tools import Document
from utils import remove_punctuation, convert_to_lower, tokenize


class CorpusAnalyzer:
    def __init__(self, corpus_path: str):
        self.corpus_fd = open(corpus_path, 'r')
        # A dictionary of documents where the keys are their identifier
        # and the value its a Document instance
        self.documents: Dict[int or str, Document] = {}
        self.stemmer = nltk.PorterStemmer()
        self.index: corpora.Dictionary = None
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
        if stemming:
            tokens = self.stemming(tokens)
        return tokens

    def stemming(self, tokens: List[str]):
        return [self.stemmer.stem(token) for token in tokens]

    def create_document_index(self, name='docs_index'):
        """This method creates a dictionary based on the taxonomy of keywords for each document."""
        docs = [d.tokens for d in self.documents.values()]
        self.index = corpora.Dictionary(docs)
        self.index.save(f'{name}.idx')

    def docs2bows(self):
        """
        Converts the document (the list of words) into the bag-of-words
        format = list of (token_id, token_count) 2-tuples.
        """
        vectors = [self.index.doc2bow(doc.tokens) for doc in self.documents.values()]
        corpora.MmCorpus.serialize('vsm_docs.mm', vectors)  # Save the corpus in the Matrix Market format
        return vectors

from typing import List


class Document:
    def __init__(self, id, tokens, title=None):
        # todo: Maybe i should keep the original text of the document for presentation
        self.id = id
        self.tokens: List[str] = tokens
        self.title = title

    def __str__(self):
        return f'{self.id}: {" ".join(self.tokens)}'

    def __repr__(self):
        return str(self)


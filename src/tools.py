from typing import List


class Document:
    def __init__(self, id, tokens):
        self.id = id
        self.tokens: List[str] = tokens

    def __str__(self):
        return f'{self.id}: {" ".join(self.tokens)}'

    def __repr__(self):
        return str(self)


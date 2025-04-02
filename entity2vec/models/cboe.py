from .entity_model import EntityModel
import numpy as np


class Cboe(EntityModel):
    def epoch(self, training_data: list[list[str]], iteration: int):
        # still need to figure out the correct way of working of cbow to
        # transfer to cboe in this context

        # ref:
        # cbow from gensim
        # https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        ...

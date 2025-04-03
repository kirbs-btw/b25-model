from .entity_model import EntityModel
import numpy as np


class Cboe(EntityModel):
    def epoch(self, training_data: list[list[str]], iteration: int):
        # still need to figure out the correct way of working of cbow to
        # transfer to cboe in this context

        # ref:
        # cbow from gensim
        # https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
        for raw_entity_set in training_data:
            # creating a list of the vectors

            entity_vectors = []
            entity_list = []
            for e in raw_entity_set:
                if not self.in_vector_map(e):
                    continue
                entity_vectors.append(self.get_vector(e))
                entity_list.append(e)

            # checking if entity set has meaning (len > 1)
            n = len(entity_vectors)
            if n < 2:
                continue
            n_minus_one = n - 1
            # ...
            # embedding lookup and average
            # output prediction
            # softmax function
            # loss (cross-entropy)

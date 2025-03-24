import pickle
import math
import numpy as np
from abc import ABC, abstractmethod

# here creating a parent class to inherit from for the other models etc

# save model
# load model
# nearest function
# but training should be different for all the models oc


class EntityModel(ABC):
    def __init__(
        self,
        training_data,
        vector_size=16,
        min_count=1,
        epochs=5,
        learning_rate=0.001,
    ):
        self.vector_size: int = vector_size
        self.min_count: int = min_count
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.__max_distance: float = math.sqrt(4 * self.vector_size)
        self.vector_map: dict = self.__create_vec_map(training_data)
        self.__train_model(training_data)

    def __create_vec_map(self, training_data: list) -> dict:
        # Flatten the nested list of clusters and extract unique entries
        unique_entries = set(text for cluster in training_data for text in cluster)
        # Create a mapping from each unique entry to a random vector
        vec_map = {
            entry: np.random.uniform(-1, 1, self.vector_size)
            for entry in unique_entries
        }
        return vec_map

    @classmethod
    @abstractmethod
    def __epoch(training_data):
        """
        here you need to implement the training of your model
        """
        pass

    def __train_model(self, training_data):
        for _ in range(self.epochs):
            self.__epoch(training_data)

    def nearest(self, word, k=1):
        # If the word isn't in the vector map, return an empty list.
        if word not in self.vector_map:
            return []

        target_vector = self.vector_map[word]

        # Cache the keys and vectors if they haven't been cached already.
        if not hasattr(self, "_words"):
            self._words = list(self.vector_map.keys())
            self._vectors = np.array(list(self.vector_map.values()))

        words = self._words
        vectors = self._vectors

        # Compute the difference between all vectors and the target.
        diff = vectors - target_vector
        # Compute squared Euclidean distances. (Avoid sqrt for every element.)
        sq_dists = np.einsum("ij,ij->i", diff, diff)

        # Exclude the target word itself by setting its distance to infinity.
        idx = words.index(word)
        sq_dists[idx] = np.inf

        # Ensure that k does not exceed the number of available words.
        k = min(k, len(sq_dists) - 1)
        # Use argpartition to get the indices of the k smallest distances.
        candidate_indices = np.argpartition(sq_dists, k)[:k]
        # Sort these indices by their squared distance.
        sorted_candidates = candidate_indices[np.argsort(sq_dists[candidate_indices])]

        # Compute the square root only for the k nearest neighbors and return the results.
        return [
            [words[i], 1 - (np.sqrt(sq_dists[i]) / self.__max_distance)]
            for i in sorted_candidates
        ]

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

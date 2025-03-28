import pickle
import math
import numpy as np
from abc import ABC, abstractmethod


class EntityModel(ABC):
    VERSION = "1.0.0"

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
        """
        create a mapping of the entities to random vectors where every value is in I=[-1,1]
        """
        assert self.min_count == 1, "min count changes are not implemented"

        # Flatten the nested list of clusters and extract unique entries
        unique_entries = set(text for cluster in training_data for text in cluster)
        # Create a mapping from each unique entry to a random vector
        vec_map = {
            entry: np.random.uniform(-1, 1, self.vector_size)
            for entry in unique_entries
        }
        return vec_map

    def get_vector(self, key: str):
        return self.vector_map[key]

    def set_vector(self, key: str, vector: list):
        self.vector_map[key] = vector

    @classmethod
    @abstractmethod
    def epoch(self, training_data: list[list[str]], iteration: int):
        """
        here you need to implement the training of your model
        this will change the vectors behind the vector_map to describe the entities
        """
        pass

    def __train_model(self, training_data: list[list[str]]):
        """
        training the model with an algorithm along the epochs
        """
        for iteration in range(self.epochs):
            self.epoch(training_data, iteration)

    def nearest(self, entity: str, k: int = 1) -> list:
        """
        getting the top k nearest vectors to an entity
        """
        assert isinstance(entity, str), "Entity must be a string"
        assert isinstance(k, int), "k must be an integer"
        assert k > 0, "k must be greater than 0"

        if entity not in self.vector_map:
            return []

        target_vector = self.vector_map[entity]

        # Cache the keys and vectors if they haven't been cached already.
        if not hasattr(self, "_entities"):
            self._entities = list(self.vector_map.keys())
            self._vectors = np.array(list(self.vector_map.values()))

        entities = self._entities
        vectors = self._vectors

        # Compute the difference between all vectors and the target.
        diff = vectors - target_vector
        # Compute squared Euclidean distances. (Avoid sqrt for every element.)
        sq_dists = np.einsum("ij,ij->i", diff, diff)

        # Exclude the target word itself by setting its distance to infinity.
        idx = entities.index(entity)
        sq_dists[idx] = np.inf

        # Ensure that k does not exceed the number of available words.
        k = min(k, len(sq_dists) - 1)
        # Use argpartition to get the indices of the k smallest distances.
        candidate_indices = np.argpartition(sq_dists, k)[:k]
        # Sort these indices by their squared distance.
        sorted_candidates = candidate_indices[np.argsort(sq_dists[candidate_indices])]

        # Compute the square root only for the k nearest neighbors and return the results.
        return [
            [entities[i], 1 - (np.sqrt(sq_dists[i]) / self.__max_distance)]
            for i in sorted_candidates
        ]

    def save(self, path):
        """
        saving the model to a pkl file
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """
        loading the model from the pkl file
        """
        with open(path, "rb") as file:
            return pickle.load(file)


### Notes for further development
# abstracting the base class even more out
# splify the nearest function to be faster and maybe also change the data type of the saved
# vectors to
# add a change vector and get vector function to prep for changes with the data type

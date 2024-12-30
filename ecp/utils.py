import pandas as pd
import numpy as np
import pickle
import math

from enum import Enum


class DistNorm(Enum):
    NONE = "none"
    L2 = "L2"


class Song2VecC:
    """
    Class to handle model training
    """

    def __init__(
        self,
        training_data,
        vector_size=16,
        window=5,
        min_count=1,
        workers=1,
        algorithm=0,
        epochs=5,
        learning_rate=0.001,
        dist_method=DistNorm.NONE,
    ):
        self.vector_size: int = vector_size
        self.window: int = window
        self.min_count: int = min_count
        self.workers: int = workers
        self.algorithm: int = algorithm
        self.epochs: int = epochs
        self.__dist_method = dist_method
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

    def __epoch(self, training_data):
        """
        faster way
        """

        for cluster in training_data:
            if len(cluster) < 2:
                continue
            base_vector = np.zeros(self.vector_size)
            cluster_size = len(cluster)
            context_size = cluster_size - 1
            for data_point in cluster:
                data_point_vector = self.vector_map[data_point]
                base_vector = np.add(data_point_vector, base_vector)

            for data_point in cluster:
                data_point_vector = self.vector_map[data_point]
                context_vector = np.divide(
                    np.subtract(base_vector, data_point_vector), context_size
                )
                dist_vector = np.subtract(context_vector, data_point_vector)
                if self.__dist_method == DistNorm.L2:
                    score = np.linalg.norm(dist_vector) / self.__max_distance
                    dist_vector * score
                # pushing it into the direction of the context_vector
                gradient = dist_vector * self.learning_rate
                updated_data_point_vector = np.add(data_point_vector, gradient)

                # changing the base vector with it to keep up with the changing context
                base_vector = np.add(base_vector, gradient)

                self.vector_map[data_point] = updated_data_point_vector

    def __train_model(self, training_data):
        for _ in range(self.epochs):
            self.__epoch(training_data)

    def nearest(self, word, k=1):
        if word not in self.vector_map:
            return []

        target_vector = self.vector_map[word]

        words = list(self.vector_map.keys())
        vectors = np.array(list(self.vector_map.values()))

        diff = vectors - target_vector
        distances = np.linalg.norm(diff, axis=1)

        idx = words.index(word)
        distances[idx] = np.inf

        k = min(k, len(distances) - 1)
        nearest_indices = np.argpartition(distances, k)[:k]

        nearest_neighbors = [(words[i], distances[i]) for i in nearest_indices]

        nearest_neighbors.sort(key=lambda x: x[1])

        return [[neighbor, dist] for neighbor, dist in nearest_neighbors]

    def nearest_k1(self, word):
        if word not in self.vector_map:
            return []

        target_vector = self.vector_map[word]
        lowest_dist_sq = np.inf
        closest_word = None

        for other_word, vector in self.vector_map.items():
            if other_word == word:
                continue  # Skip the word itself

            # Compute squared Euclidean distance
            diff = vector - target_vector
            dist_sq = np.dot(diff, diff)

            if dist_sq < lowest_dist_sq:
                lowest_dist_sq = dist_sq
                closest_word = other_word

        if closest_word is not None:
            # If you need the actual Euclidean distance
            lowest_dist = np.sqrt(lowest_dist_sq)
            return [(closest_word, lowest_dist)]
        else:
            return []

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

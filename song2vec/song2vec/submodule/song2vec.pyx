import pandas as pd
import numpy as np
cimport numpy as np

print("fkjkj")

cdef class Song2Vec:
    '''
    Class to handle model training
    '''

    cdef int vector_size
    cdef int window
    cdef int min_count
    cdef int workers
    cdef int algorithm
    cdef int epochs
    cdef float learning_rate
    cdef dict vector_map
    

    def __init__(self, list training_data, int vector_size=16, int window=5, int min_count=1, int workers=1, int algorithm=0, int epochs=5, float learning_rate=0.01):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.algorithm = algorithm
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vector_map = self.__create_vec_map(training_data)

        
    def __create_vec_map(self, list training_data) -> dict:
        cdef list unique_entries = []
        cdef dict vec_map = {}
        cdef np.ndarray[np.float64_t, ndim=1] random_vector
        cdef str text
        cdef int i

        # Flatten the nested list and add unique entries
        for cluster in training_data:
            for text in cluster:
                if text not in unique_entries:
                    unique_entries.append(text)

        # Create a random vector for each unique entry
        for entry in unique_entries:
            random_vector = np.random.uniform(-1, 1, self.vector_size)
            vec_map[entry] = random_vector

        return vec_map
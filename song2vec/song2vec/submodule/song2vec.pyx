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
    cdef dict song_vectors
    cdef dict context_vectors
    
    def __init__(self, list training_data, int vector_size=16, int window=5, int min_count=1, int workers=1, int algorithm=0, int epochs=5, float learning_rate=0.01):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.algorithm = algorithm
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        print("lfkjsklfj")
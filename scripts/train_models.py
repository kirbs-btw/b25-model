from gensim.models import Word2Vec
import pickle
import os
import sys

sys.path.append(os.path.abspath("../"))

from sge import *
from cboe import *

with open("../data/tokenized_data/playlist_names/dataset_train_v3.pkl", "rb") as f:
    train_dataset = pickle.load(f)


algorithms_map = {0: "CBOW", 1: "SG"}

none_window_algorithms_map = {"ECP": "ECP", "SGE": "SGE", "CBOE": "CBOE"}

window_sizes = [10, 150]
epochs = [5, 20]
vector_sizes = [64, 256, 512]

for algorithm in algorithms_map:
    for vector_size in vector_sizes:
        for epoch in epochs:
            for window_size in window_sizes:
                model_name = f"b25-{algorithms_map[algorithm]}-{vector_size}-{epoch}-{window_size}"
                model_save_path = f"../models_str/{model_name}.model"

                print(f"working on: {model_name}")
                # checking if the model exists to not train it again
                if os.path.exists(model_save_path):
                    print(f"{model_save_path} exists. Skipping...")
                    continue

                if algorithm == 0:
                    model = Word2Vec(
                        sentences=train_dataset,
                        workers=10,
                        vector_size=vector_size,
                        window=window_size,
                        min_count=1,
                        sg=algorithm,
                        epochs=epoch,
                    )
                elif algorithm == 1:
                    model = Word2Vec(
                        sentences=train_dataset,
                        workers=10,
                        vector_size=vector_size,
                        window=window_size,
                        min_count=1,
                        sg=algorithm,
                        ns_exponent=0.0,
                        epochs=epoch,
                    )
                model.save(model_save_path)

# for algorithm in none_window_algorithms_map:
#     for vector_size in vector_sizes:
#         for epoch in epochs:
#             model_name = (
#                 f"b25-{none_window_algorithms_map[algorithm]}-{vector_size}-{epoch}-inf"
#             )
#             model_save_path = f"../models_str/{model_name}.pkl"

#             if os.path.exists(model_save_path):
#                 print(f"{model_save_path} exists. Skipping...")
#                 continue

#             if algorithm == "ECP":
#                 model = son2vecgl_c.Song2GloveC(
#                     training_data=train_dataset,
#                     vector_size=vector_size,
#                     epochs=epoch,
#                     learning_rate=0.025,
#                 )
#                 with open(model_save_path, "wb") as f:
#                     pickle.dump(model, f)

#             elif algorithm == "SGE":
#                 model = SkipGramEntity2Vec(
#                     tokenized_data=train_dataset,
#                     vector_size=vector_size,
#                     min_count=1,
#                     epochs=epoch,
#                     learning_rate=0.025,
#                 )
#                 model.save(model_save_path)
#             elif algorithm == "CBOE":
#                 model = Entity2Vec(
#                     sentences=train_dataset,
#                     vector_size=vector_size,
#                     min_count=1,
#                     window=0,
#                     epochs=epoch,
#                     learning_rate=0.025,
#                 )
#                 model.save(model_save_path)

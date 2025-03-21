from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import os
import csv
from enum import Enum

with open("../data/tokenized_data/playlist_names/dataset_test_v3.pkl", "rb") as f:
    tokenized_playlists = pickle.load(f)


test_set = tokenized_playlists[:250]

TOP_N = 250


class ModelType(Enum):
    GENSIM = "gensim"
    SGE = "sge"
    CBOE = "cboe"
    ECP = "ecp"


def get_nearest(model, song: str, top_n: int, model_type: ModelType):
    """
    using the embedding model to get the same output every time and sorting out the
    picking of the model
    """
    if model_type == ModelType.GENSIM:
        return model.wv.most_similar(song, topn=top_n)
    elif model_type == ModelType.CBOE:
        raise NotImplementedError("CBOE not implemented")
    elif model_type == ModelType.ECP:
        raise NotImplementedError("ECP not implemented")
    elif model_type == ModelType.SGE:
        raise NotImplementedError("SGE not implemented")
    else:
        raise TypeError("this type is not known jet")


def evaluate_recall_precision_macro(
    model, playlists, top_n=10, model_type=ModelType.GENSIM
):
    """
    Evaluates the model by computing the macro-average precision and recall.
    For each song (query), the ground truth is all the other songs in its playlist.
    """
    total_precision = 0.0
    total_recall = 0.0
    valid_queries = 0

    for playlist in playlists:
        for song in playlist:
            ground_truth = set(playlist) - {song}
            if not ground_truth:
                continue
            try:
                similar_songs = get_nearest(model, song, top_n, model_type)
            except KeyError:
                continue
            recommended = {rec_song for rec_song, _ in similar_songs}
            correct = recommended.intersection(ground_truth)

            precision = len(correct) / top_n
            recall = len(correct) / len(ground_truth)

            total_precision += precision
            total_recall += recall
            valid_queries += 1

    avg_precision = total_precision / valid_queries if valid_queries else 0
    avg_recall = total_recall / valid_queries if valid_queries else 0
    return avg_precision, avg_recall


def evaluate_recall_precision_micro(
    model, playlists, top_n=100, model_type=ModelType.GENSIM
):
    """
    Computes micro-averaged precision and recall over all queries.
    """
    total_correct = 0
    total_recommended = 0
    total_relevant = 0

    for playlist in playlists:
        for song in playlist:
            ground_truth = set(playlist) - {song}
            if not ground_truth:
                continue

            try:
                similar_words = get_nearest(model, song, top_n, model_type)

                similar_songs = [
                    (word, sim) for word, sim in similar_words if sim >= 0.75
                ]
            except KeyError:
                continue

            recommended = {rec_song for rec_song, _ in similar_songs}
            correct = recommended.intersection(ground_truth)

            total_correct += len(correct)
            total_recommended += top_n
            total_relevant += len(ground_truth)

    precision = total_correct / total_recommended if total_recommended else 0
    recall = total_correct / total_relevant if total_relevant else 0
    return precision, recall


# precisssion@1
def precision_at_1(model, model_type=ModelType.GENSIM):
    tested = 0
    correct = 0

    test_set = tokenized_playlists[:250]

    vgl_a = 0
    vgl_b = 0

    for playlist in test_set:
        for song in playlist:
            vgl_a += 1

            try:

                similar_words = get_nearest(model, song, 1, model_type)
                if similar_words == []:
                    continue

                tested += 1

                if any(word[0] in playlist for word in similar_words):
                    correct += 1
            except:
                vgl_b += 1
                continue

    return correct / tested


if __name__ == "__main__":
    algorithms_map = {1: "SG"}
    window_sizes = [10, 150]
    epochs = [5, 20]
    vector_sizes = [256, 512]

    csv_file = "models_benchmark.csv"
    csv_header = [
        "Model Name",
        "Algorithm",
        "Vector Size",
        "Epochs",
        "Window Size",
        "Precision",
        "Recall",
        "F1 Score",
        "Precision@1",
    ]

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(csv_header)

    for algorithm in algorithms_map:
        for vector_size in vector_sizes:
            for epoch in epochs:
                for window_size in window_sizes:
                    model_name = f"b25-{algorithms_map[algorithm]}-{vector_size}-{epoch}-{window_size}"
                    model_save_path = f"../models_str/{model_name}.model"

                    if not os.path.exists(model_save_path):
                        continue
                    print(f"Testing {model_name}")
                    model = Word2Vec.load(model_save_path)

                    micro_precision, micro_recall = evaluate_recall_precision_micro(
                        model, test_set, top_n=TOP_N
                    )
                    F1_micro = 2 * (
                        (micro_precision * micro_recall)
                        / (micro_recall + micro_precision)
                    )
                    precision_at_one = precision_at_1(model)

                    writer.writerow(
                        [
                            model_name,
                            algorithms_map[algorithm],
                            vector_size,
                            epoch,
                            window_size,
                            round(micro_precision, 4),
                            round(micro_recall, 4),
                            round(F1_micro, 4),
                            round(precision_at_one, 4),
                        ]
                    )

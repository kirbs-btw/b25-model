import numpy as np
import pickle
from collections import Counter, defaultdict
import math
import concurrent.futures


class SkipGramEntity2Vec:
    def __init__(
        self,
        tokenized_data,
        vector_size=2048,
        min_count=1,
        epochs=5,
        learning_rate=0.01,
        num_workers=10,  # Number of CPU cores to use
    ):
        """
        Initialisiert und trainiert ein sehr einfaches Skip-Gram-Modell.
        """
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.lr = learning_rate
        self.num_workers = num_workers  # Added to configure parallelism

        # 1) Wortschatz aufbauen
        self._build_vocab(tokenized_data)

        # Precompute index lists for each sentence (nur Wörter im Vokabular)
        self.sentences_indices = [
            [self.word_to_idx[w] for w in sent if w in self.word_to_idx]
            for sent in tokenized_data
            if len([w for w in sent if w in self.word_to_idx]) >= 2
        ]

        # 2) Gewichte initialisieren
        self.W_in = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
        ) / vector_size
        self.W_out = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
        ) / vector_size

        # 3) Modell trainieren (Skip-Gram)
        self._train()

    def _train(self):
        for epoch in range(self.epochs):
            total_loss = 0.0
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                # Parallelize training by dispatching work across multiple workers
                futures = [
                    executor.submit(self._train_sentence, word_indices)
                    for word_indices in self.sentences_indices
                ]
                for future in concurrent.futures.as_completed(futures):
                    total_loss += future.result()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def _train_sentence(self, word_indices):
        """
        Train a single sentence in parallel.
        This method is used for parallel processing of each sentence in the dataset.
        """
        sentence_loss = 0.0
        for i, target_idx in enumerate(word_indices):
            # Alle Kontext-Wort-Indizes außer dem Target
            context_indices = word_indices[:i] + word_indices[i + 1 :]

            if not context_indices:
                continue

            # Vektor für das target Wort
            target_vec = self.W_in[target_idx]  # shape: (vector_size,)
            # Vektoren für alle Kontext-Wörter
            context_vecs = self.W_out[
                context_indices
            ]  # shape: (n_context, vector_size)

            # Compute dot products for all context words at once
            scores = np.dot(context_vecs, target_vec)  # shape: (n_context,)
            pred_probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid activation

            # Verlust summieren (log loss)
            sentence_loss += -np.sum(np.log(pred_probs + 1e-10))

            # Gradient berechnen (dL/d(score))
            grads = pred_probs - 1.0  # shape: (n_context,)

            # Update: Vektorized Ausgabe-Embeddings
            self.W_out[context_indices] -= self.lr * np.outer(grads, target_vec)
            # Update: Input-Embedding: Summe der Gradienten von allen Kontext-Wörtern
            self.W_in[target_idx] -= self.lr * np.dot(grads, context_vecs)

        return sentence_loss

    def _build_vocab(self, sentences):
        """
        Baut das Vokabular (word_to_idx / idx_to_word) auf und
        entfernt seltene Wörter anhand von min_count.
        """
        word_counts = Counter()
        for sent in sentences:
            word_counts.update(sent)

        # filter words sub min count
        self.vocab = [w for w, c in word_counts.items() if c >= self.min_count]

        # (optional) result reproduction
        self.vocab = sorted(self.vocab)

        # entity <-> id mapping
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def nearest(self, word_or_vector, k=1):
        """
        Findet die k nächsten Wörter/Entitäten nach Cosine-Similarity.

        Argumente:
            word_or_vector (Union[str, np.ndarray]): Entweder ein Wort (dann wird dessen Embedding genommen)
                                                     oder ein Vektor.
            k (int): Anzahl der nächsten Nachbarn.

        Rückgabe:
            Liste von (Wort, Similarity)-Tupeln.
        """
        if isinstance(word_or_vector, str):
            # Eingegebenes Wort in Vektor umwandeln
            if word_or_vector not in self.word_to_idx:
                raise ValueError(f"Unbekanntes Wort/Entität: {word_or_vector}")
            vector = self.W_in[self.word_to_idx[word_or_vector]]
        else:
            # Wir nehmen an, es ist bereits ein Vektor
            vector = word_or_vector

        # Eingabevektor normalisieren
        norm_vector = vector / (np.linalg.norm(vector) + 1e-10)

        # Alle W_in-Embedding-Vektoren normalisieren
        W_in_norm = self.W_in / (
            np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        )

        # Cosine-Similarity berechnen
        similarities = np.dot(W_in_norm, norm_vector)

        # Sortiere absteigend nach Similarity
        nearest_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in nearest_indices:
            w = self.idx_to_word[idx]
            if w == word_or_vector:
                continue
            results.append((w, similarities[idx]))
            if len(results) == k:
                break

        return results

    def save(self, path):
        """
        Speichere das Modell auf die Festplatte mittels pickle.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vector_size": self.vector_size,
                    "word_to_idx": self.word_to_idx,
                    "idx_to_word": self.idx_to_word,
                    "W_in": self.W_in,
                    "W_out": self.W_out,
                },
                f,
            )
        print(f"Modell gespeichert unter: {path}")

    @classmethod
    def load(cls, path):
        """
        Lade Modell von der Festplatte.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Leere Instanz erzeugen
        model = cls(
            sentences=[],
            vector_size=data["vector_size"],
            min_count=1,
            window=100,
            epochs=0,  # Kein erneutes Training
            learning_rate=0.0,
        )
        model.word_to_idx = data["word_to_idx"]
        model.idx_to_word = data["idx_to_word"]
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model


if __name__ == "__main__":
    # Beispiel-Datensatz
    train_dataset = [
        ["john", "loves", "mary"],
        ["mary", "loves", "chocolate"],
        ["dog", "bites", "man"],
        ["man", "bites", "dog"],
    ]

    model = SkipGramEntity2Vec(
        tokenized_data=train_dataset,
        vector_size=8,
        min_count=1,
        epochs=10,
        learning_rate=0.01,
        num_workers=10,  # Set number of workers
    )

    print("\nNearest to 'john':", model.nearest("john", k=2))
    print("Nearest to 'dog':", model.nearest("dog", k=2))

    model.save("tmp/demo_skipgram_model.pkl")
    loaded_model = SkipGramEntity2Vec.load("demo_skipgram_model.pkl")

    print(
        "\nNearest to 'chocolate' (from loaded model):",
        loaded_model.nearest("chocolate", k=2),
    )

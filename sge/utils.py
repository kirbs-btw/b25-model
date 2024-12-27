import numpy as np
import pickle
from collections import Counter, defaultdict
import math


class SkipGramEntity2Vec:
    def __init__(
        self,
        tokenized_data,
        vector_size=2048,
        min_count=1,
        epochs=5,
        learning_rate=0.01,
    ):
        """
        Initialisiert und trainiert ein sehr einfaches Skip-Gram-Modell.

        Argumente:
            sentences (List[List[str]]): Trainingsdaten (Liste von Listen von Wörtern/Entitäten).
            vector_size (int): Dimension der Embeddings.
            min_count (int): Ignoriere Wörter/Entitäten mit Häufigkeit < min_count..
            epochs (int): Anzahl Epochen.
            learning_rate (float): Lernrate.
        """
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.lr = learning_rate

        # 1) Wortschatz aufbauen
        self._build_vocab(tokenized_data)

        # 2) Gewichte initialisieren
        #    - W_in: Embeddings für Input (target-Wort)
        #    - W_out: Embeddings für Output (Kontext-Wort)
        self.W_in = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
        ) / vector_size
        self.W_out = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
        ) / vector_size

        # 3) Modell trainieren (Skip-Gram)
        self._train(tokenized_data)

    def _build_vocab(self, sentences):
        """
        Baut das Vokabular (word_to_idx / idx_to_word) auf und
        entfernt seltene Wörter anhand von min_count.
        """
        word_counts = Counter()
        for sent in sentences:
            word_counts.update(sent)

        # Filtere Wörter unterhalb von min_count
        self.vocab = [w for w, c in word_counts.items() if c >= self.min_count]
        # Für Reproduzierbarkeit sortieren (optional)
        self.vocab = sorted(self.vocab)

        # Mapping Wörter <-> IDs
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def _train(self, sentences):
        for epoch in range(self.epochs):
            total_loss = 0.0
            for sent in sentences:
                # Satz in Indizes umwandeln, unbekannte Wörter ignorieren
                word_indices = [
                    self.word_to_idx[w] for w in sent if w in self.word_to_idx
                ]
                if len(word_indices) < 2:
                    continue

                # Skip-Gram: Für jedes Wort als "target" werden alle anderen Satz-Wörter als Kontext genommen
                for i, target_idx in enumerate(word_indices):
                    context_indices = word_indices[:i] + word_indices[i + 1 :]
                    if not context_indices:
                        continue

                    # Für jeden Kontext im Satz
                    for context_idx in context_indices:
                        # Forward Pass
                        # score = W_in[target_idx] · W_out[context_idx]
                        score = np.dot(self.W_in[target_idx], self.W_out[context_idx])

                        # Sigmoid / "Binary Cross-Entropy"-like Loss (keine negativen Beispiele hier)
                        pred_prob = 1.0 / (1.0 + np.exp(-score))
                        loss = -math.log(
                            pred_prob + 1e-10
                        )  # log(prob) des korrekten Wortes
                        total_loss += loss

                        # Backprop
                        grad = pred_prob - 1.0  # dL/d(score)

                        # Update Output-Embedding
                        self.W_out[context_idx] -= (
                            self.lr * grad * self.W_in[target_idx]
                        )

                        # Update Input-Embedding
                        self.W_in[target_idx] -= (
                            self.lr * grad * self.W_out[context_idx]
                        )

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

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
    )

    print("\nNearest to 'john':", model.nearest("john", k=2))
    print("Nearest to 'dog':", model.nearest("dog", k=2))

    model.save("tmp/demo_skipgram_model.pkl")
    loaded_model = SkipGramEntity2Vec.load("demo_skipgram_model.pkl")

    print(
        "\nNearest to 'chocolate' (from loaded model):",
        loaded_model.nearest("chocolate", k=2),
    )

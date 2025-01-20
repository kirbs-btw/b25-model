import numpy as np
import pickle
from collections import Counter
import math


class Entity2Vec:
    def __init__(
        self,
        sentences,
        vector_size=512,
        min_count=1,
        window=100,  # In this simplified version, still for clarity
        workers=1,  # Not used in this naive example
        epochs=5,
        learning_rate=0.01,
    ):
        """
        Initialize and train a simple CBOW-like embedding model.

        Arguments:
            sentences (list of lists of str): The training data, each element is a list of words/entities.
            vector_size (int): Dimensionality of the embeddings.
            min_count (int): Ignore all words/entities with total frequency lower than this.
            window (int): Context window size. For demonstration, we treat the entire sentence as context.
            workers (int): Number of worker threads. (Not used in naive version)
            epochs (int): Number of passes (epochs) over the training data.
            learning_rate (float): Learning rate for gradient updates.
        """
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.lr = learning_rate

        # 1) Build the vocabulary
        self._build_vocab(sentences)

        # 2) Initialize weights
        # We'll have two embedding matrices:
        #    - W_in for input (context) embeddings
        #    - W_out for output (target) embeddings
        vocab_size = len(self.idx_to_word)
        self.W_in = (
            np.random.rand(vocab_size, vector_size).astype(np.float32) - 0.5
        ) / vector_size
        self.W_out = (
            np.random.rand(vocab_size, vector_size).astype(np.float32) - 0.5
        ) / vector_size

        # 3) Train the model
        self._train(sentences)

    def _build_vocab(self, sentences):
        """
        Builds vocab from the input sentences and sets up
        word_to_idx / idx_to_word mappings, ignoring words
        with frequency < min_count.
        """
        word_counts = Counter()
        for sent in sentences:
            word_counts.update(sent)

        # Filter words by min_count
        self.vocab = [w for w, c in word_counts.items() if c >= self.min_count]
        self.vocab = sorted(self.vocab)  # sort for reproducibility

        # Create mapping dictionaries
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def _train(self, sentences):
        """
        Train the model using a naive CBOW approach.
        The entire sentence is considered the context (excluding the target).
        """
        for epoch in range(self.epochs):
            total_loss = 0.0
            for sent in sentences:
                # Convert sentence words to indices (filter out unknown words)
                word_indices = [
                    self.word_to_idx[w] for w in sent if w in self.word_to_idx
                ]
                n = len(word_indices)
                if n < 2:
                    continue  # Skip short sentences

                # --- Optimization key: sum of all embeddings in the sentence ---
                # We'll subtract the target's embedding to get the context sum.
                sent_sum = np.sum(
                    self.W_in[word_indices], axis=0
                )  # shape: (vector_size,)

                for i, target_idx in enumerate(word_indices):
                    # The context sum is (sent_sum - embedding_of_target).
                    # The context vector is that difference, divided by (n - 1).
                    context_vec = (sent_sum - self.W_in[target_idx]) / (n - 1)

                    # Forward pass:
                    score = np.dot(context_vec, self.W_out[target_idx])  # scalar
                    pred_prob = 1.0 / (1.0 + np.exp(-score))
                    loss = -math.log(pred_prob + 1e-10)
                    total_loss += loss

                    # Backprop (binary cross-entropy style):
                    grad = pred_prob - 1.0  # dL/dscore

                    # Update W_out for the target word:
                    #   W_out[target_idx] -= lr * grad * context_vec
                    self.W_out[target_idx] -= self.lr * grad * context_vec

                    # Update W_in for *all context words* in one shot.
                    # context_indices = all but `target_idx`.
                    # We'll apply the same gradient to each context word embedding:
                    #   dL/dW_in[context] = grad * W_out[target_idx] / (n - 1)
                    grad_context = grad * self.W_out[target_idx] / (n - 1)

                    # Instead of looping, we can do boolean indexing or slicing:
                    # But we need to exclude the target word from the update.
                    # We'll do a quick trick: update *all* then add back the target since we subtracted it by accident.

                    self.W_in[word_indices] -= self.lr * grad_context
                    # We added an unwanted update to the target word (target_idx),
                    # so we *undo* that for the target word:
                    self.W_in[target_idx] += self.lr * grad_context

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def nearest(self, word_or_vector, k=1):
        """
        Find the k nearest words/entities (by cosine similarity).
        If `word_or_vector` is a string, we look up its embedding.
        If `word_or_vector` is a vector (np.ndarray), we use it directly.

        Returns:
            List of tuples (word, similarity).
        """

        # 1) If the input is a string, get its embedding:
        if isinstance(word_or_vector, str):
            if word_or_vector not in self.word_to_idx:
                raise ValueError(f"Unknown word/entity: {word_or_vector}")
            vector = self.W_in[self.word_to_idx[word_or_vector]]
            skip_index = self.word_to_idx[word_or_vector]
        else:
            vector = word_or_vector
            skip_index = None

        norm_vector = vector / (np.linalg.norm(vector) + 1e-10)

        if not hasattr(self, "W_in_norm"):
            self.W_in_norm = self.W_in / (
                np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
            )

        similarities = np.dot(self.W_in_norm, norm_vector)
        max_indices = np.argpartition(similarities, -(k + 1))[-(k + 1) :]

        max_indices = max_indices[np.argsort(similarities[max_indices])[::-1]]

        results = []
        for idx in max_indices:
            if idx == skip_index:
                continue
            results.append((self.idx_to_word[idx], similarities[idx]))
            if len(results) == k:
                break

        return results

    def save(self, path):
        """
        Save model to disk using pickle.
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
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """
        Load model from disk.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Create an empty instance
        model = cls(
            sentences=[],
            vector_size=data["vector_size"],
            min_count=1,
            window=100,
            workers=1,
            epochs=0,
            learning_rate=0.0,
        )
        model.word_to_idx = data["word_to_idx"]
        model.idx_to_word = data["idx_to_word"]
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model


if __name__ == "__main__":
    train_dataset = [
        ["john", "loves", "mary"],
        ["mary", "loves", "chocolate"],
        ["dog", "bites", "man"],
        ["man", "bites", "dog"],
    ]

    model = Entity2Vec(
        sentences=train_dataset,
        vector_size=8,
        min_count=1,
        window=100,
        epochs=10,
        learning_rate=0.01,
    )

    print("\nNearest to 'john':", model.nearest("john", k=2))
    print("Nearest to 'dog':", model.nearest("dog", k=2))

    model.save("demo_entity2vec.pkl")
    loaded_model = Entity2Vec.load("demo_entity2vec.pkl")

    print(
        "\nNearest to 'chocolate' (from loaded model):",
        loaded_model.nearest("chocolate", k=2),
    )

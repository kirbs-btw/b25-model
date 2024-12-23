import numpy as np
import pickle
from collections import Counter, defaultdict
import math


class Entity2Vec:
    def __init__(
        self,
        sentences,
        vector_size=512,
        min_count=1,
        window=100,  # In this simplified version, we ignore window except for clarity
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
            window (int): Context window size.  For demonstration, we treat the entire sentence as context.
            workers (int): Number of worker threads. (Not implemented in naive version)
            epochs (int): Number of passes (epochs) over the training data.
            learning_rate (float): Learning rate for gradient updates.
        """
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.lr = learning_rate

        # 1) Build the vocabulary and filter words based on min_count
        self._build_vocab(sentences)

        # 2) Initialize weights
        #    We'll have two embedding matrices:
        #    - W_in for input (context) embeddings
        #    - W_out for output (target) embeddings
        self.W_in = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
        ) / vector_size
        self.W_out = (
            np.random.rand(len(self.idx_to_word), vector_size) - 0.5
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

        # Sort vocab so it’s reproducible (optional)
        self.vocab = sorted(self.vocab)

        # Create mapping dictionaries
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def _train(self, sentences):
        """
        Train the model using a naive CBOW approach.  The entire
        sentence is considered the context (no weighting).
        """
        for epoch in range(self.epochs):
            total_loss = 0.0
            for sent in sentences:
                # Convert sentence words to indices (filter out unknown words)
                word_indices = [
                    self.word_to_idx[w] for w in sent if w in self.word_to_idx
                ]
                if len(word_indices) < 2:
                    continue  # Skip short sentences

                # For each target word, treat *all other words* in the sentence as context
                for i, target_idx in enumerate(word_indices):
                    context_indices = word_indices[:i] + word_indices[i + 1 :]
                    if not context_indices:
                        continue

                    # CBOW context vector: average of W_in over context
                    context_vec = np.mean(self.W_in[context_indices], axis=0)

                    # Forward pass: predict the target_idx
                    # Score = context_vec dot W_out[target_idx]
                    score = np.dot(context_vec, self.W_out[target_idx])
                    # We'll do a simple softmax against just the target word for demonstration
                    # For a real-world scenario, you would do negative sampling or full softmax across all vocab
                    # Probability of the correct word = sigmoid(score)
                    # We'll do a binary cross-entropy style update (like negative sampling with 1 negative)
                    pred_prob = 1.0 / (1.0 + np.exp(-score))

                    # Loss = -(log(pred_prob)) for the correct word
                    # We'll keep track of the loss just for monitoring
                    loss = -math.log(pred_prob + 1e-10)
                    total_loss += loss

                    # Backprop
                    # gradient wrt score = (pred_prob - 1)
                    grad = pred_prob - 1.0

                    # Update W_out for the target
                    self.W_out[target_idx] -= self.lr * grad * context_vec

                    # Update W_in for each context word
                    grad_context = grad * self.W_out[target_idx]
                    # Distribute the gradient across context embeddings
                    for c_i in context_indices:
                        self.W_in[c_i] -= self.lr * grad_context / len(context_indices)

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def nearest(self, word_or_vector, k=1):
        """
        Find the k nearest words/entities (by cosine similarity).
        If `word_or_vector` is a string, we look up its embedding.
        If `word_or_vector` is a vector (np.ndarray), we use it directly.

        Returns:
            List of tuples (word, similarity).
        """
        if isinstance(word_or_vector, str):
            # Convert word to its embedding
            if word_or_vector not in self.word_to_idx:
                raise ValueError(f"Unknown word/entity: {word_or_vector}")
            vector = self.W_in[self.word_to_idx[word_or_vector]]
        else:
            # Assume it's already a vector
            vector = word_or_vector

        # Normalize the input vector
        norm_vector = vector / (np.linalg.norm(vector) + 1e-10)

        # Compute cosine similarity with all embeddings
        W_in_norm = self.W_in / (
            np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(W_in_norm, norm_vector)

        # Get top k (excluding the queried word if it's in the vocab)
        # argsort in descending order:
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
            min_count=1,  # Not strictly necessary to match
            window=100,
            workers=1,
            epochs=0,  # We won't train during load
            learning_rate=0.0,
        )
        model.word_to_idx = data["word_to_idx"]
        model.idx_to_word = data["idx_to_word"]
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model


if __name__ == "__main__":
    # Example usage:
    # Suppose we have a small corpus
    train_dataset = [
        ["john", "loves", "mary"],
        ["mary", "loves", "chocolate"],
        ["dog", "bites", "man"],
        ["man", "bites", "dog"],
    ]

    # Train our Entity2Vec model
    model = Entity2Vec(
        sentences=train_dataset,
        vector_size=8,
        min_count=1,
        window=100,
        epochs=10,
        learning_rate=0.01,
    )

    # Test nearest words
    print("\nNearest to 'john':", model.nearest("john", k=2))
    print("Nearest to 'dog':", model.nearest("dog", k=2))

    # Save & load
    model.save("demo_entity2vec.pkl")
    loaded_model = Entity2Vec.load("demo_entity2vec.pkl")

    print(
        "\nNearest to 'chocolate' (from loaded model):",
        loaded_model.nearest("chocolate", k=2),
    )

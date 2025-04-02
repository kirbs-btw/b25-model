from .entity_model import EntityModel
import numpy as np
import math


class Sge(EntityModel):
    def epoch(self, training_data: list[list[str]], iteration: int):
        """
        implements Skip-Gram Training with an infinite Context and weights
        for non sequential data training
        """
        for raw_entity_set in training_data:
            set_loss = 0
            # obey min_count - if min count =1 -> entity_set == raw_entity_set
            entity_set = [
                entity for entity in raw_entity_set if self.in_vector_map(entity)
            ]

            if len(entity_set) < 2:
                continue

            for idx, target_entity in enumerate(entity_set):
                context_entities = entity_set[:idx] + entity_set[idx + 1 :]
                target_vec = self.get_vector(target_entity)
                context_vecs = [
                    self.get_vector(context_entity)
                    for context_entity in context_entities
                ]

                scores = np.dot(context_vecs, target_vec)
                pred_probs = 1.0 / (1.0 + np.exp(-scores))

                set_loss += -np.sum(np.log(pred_probs + 1e-10))

                grads = pred_probs - 1.0

                for context_entity, grad in zip(context_entities, grads):
                    self.set_vector(
                        context_entity,
                        self.get_vector(context_entity)
                        - self.learning_rate * grad * target_vec,
                    )

                # Update: Input-Embedding des Ziel-Entities (Summe der Gradienten × Kontext-Vektoren)
                self.set_vector(
                    target_entity,
                    self.get_vector(context_entity)
                    - self.learning_rate * np.dot(grads, context_vecs),
                )


if __name__ == "__main__":
    data = [
        ["hello", "i", "a"],
        ["i", "a", "fff"],
        ["i", "hello"],
        ["i", "hello"],
        ["i", "hello"],
    ]

    model = Sge(
        training_data=data, vector_size=16, min_count=1, epochs=5, learning_rate=0.025
    )

    print(model.nearest("hello"))

    assert isinstance(model.nearest("hello"), list)


### dev notes
## still need unit testing to get it clear
## also a review of the sg process mapped to this code...

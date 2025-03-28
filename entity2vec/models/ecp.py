from .entity_model import EntityModel
import numpy as np
import math


class Ecp(EntityModel):
    def epoch(self, training_data: list[list[str]], iteration: int):
        for raw_entity_set in training_data:
            # creating a list of the vectors

            entity_vectors = []
            entity_list = []
            for e in raw_entity_set:
                if not self.in_vector_map(e):
                    continue
                entity_vectors.append(self.get_vector[e])
                entity_list.append(e)

            # checking if entity set has meaning (len > 1)
            n = len(entity_vectors)
            if n < 2:
                continue
            n_minus_one = n - 1

            vector_sum = np.sum(entity_vectors, axis=0)

            for i, target_vector in enumerate(entity_vectors):
                context_vector = (vector_sum - target_vector) / n_minus_one

                # distance of the target and the context
                score = np.dot(context_vector, target_vector)
                pred_prob = 1.0 / (1.0 + np.exp(-score))  # value between 1 and 0.5

                gradient_vector = context_vector - target_vector

                scaled_gradient_vector = (
                    gradient_vector * self.learning_rate * pred_prob
                )
                self.set_vector(entity_list[i], target_vector + scaled_gradient_vector)
                vector_sum += scaled_gradient_vector


if __name__ == "__main__":
    data = [["hello", "i", "a"], ["i", "a", "fff"], ["i", "hello"]]

    model = Ecp(
        training_data=data, vector_size=16, min_count=1, epochs=5, learning_rate=0.025
    )

    assert isinstance(model.nearest("hello"), list)

### Development Notes
### looking into backpropagation
### writing test for every model

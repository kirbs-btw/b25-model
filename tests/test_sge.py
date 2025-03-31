from entity2vec import Sge


def test_training():
    data = [["hello", "i", "a"], ["i", "a", "fff"], ["i", "hello"]]

    model = Sge(
        training_data=data, vector_size=16, min_count=1, epochs=5, learning_rate=0.025
    )

    assert isinstance(model.nearest("hello"), list)

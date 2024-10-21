# b25

## General 
Training a standard word embedding model involves exposing it to a large corpus of sentences, allowing it to learn the semantic relationships between words by analyzing how they co-occur within sentence structures. This process relies on the distributional hypothesis, which suggests that words appearing in similar contexts tend to have similar meanings.

The approach in question extends this concept by drawing an analogy between sentences as collections of words and playlists as collections of songs. To construct a model for playlists, one could initially consider the song titles as the equivalent of words and the playlists as analogous to sentences. However, this direct mapping poses challenges when utilizing algorithms designed for natural language processing.

Many word embedding algorithms, such as Word2Vec, focus on local context windows—capturing the relationships between a word and its immediate neighbors. This assumption of locality doesn't directly translate to playlists, where the relationship between songs isn't necessarily dependent on their proximity or order within the playlist. Hence, adapting such algorithms for playlist modeling requires addressing the contextual difference between word co-occurrence and song co-inclusion in playlists. 

## Data
The base dataset is from [kaggle](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)

## Models
### Text embedding Models
These models are able to create vectors out of song names.

- b25-sn-v50: song name - Vector dim 50

- b25-sn-v256: song name - Vector dim 256

## Model Validation

The dataset is divided into two sets: training and testing, to mitigate overfitting. The model evaluation involves selecting a song from a playlist and allowing the model to predict the next song. If the predicted song is present in the same playlist, the model is marked as correct. Currently, the model is not evaluated on its ability to understand the context or sequence of songs, which represents a potential future testing phase. In essence, the model receives one song as input and suggests the next best matching song based on its learned associations.

### Model Evaluation Report

#### Model 1: `b25-sn-v256`
- **Description**: This model is trained using the base dataset with no modifications or preprocessing, utilizing the most basic version of the model.
- **Accuracy**: `0.0739` over 100 playlists.
- **Remarks**: 
  - The accuracy is relatively low, which suggests the model’s song suggestions are not closely aligned with user preferences.
  - There is potential for improvement with further optimization, though it’s uncertain if the accuracy will improve with more playlist data.

#### Model 2: `b25-sn-v256-b`
- **Description**: This model is trained on a larger dataset. The base dataset is extended with shuffled playlists to add variability.
- **Accuracy**: `0.2774` over 100 playlists.
- **Remarks**:
  - Achieving a ~27.7% accuracy means that approximately 1 in 4 suggested songs is correct, which is a solid improvement compared to the first model.
  - A decent result for a second iteration, indicating some improvements from increased data variety.

#### Model 3: `b25-sn-v256-c`
- **Description**: Based on the dataset of `b25-sn-v256-b`, but with an increased context window of 20 (instead of the previous 5).
- **Accuracy**: `0.4882` over 100 playlists.
- **Remarks**: 
  - A significant improvement in accuracy (~48.8%), suggesting that increasing the context window allows for better understanding of user preferences and enhances the quality of song recommendations.

#### Model 4: `b25-sn-v256-d`
- **Description**: This model uses the same dataset as `b25-sn-v256-c`, but the training set is reduced to 80% of the data to avoid overfitting and allow for better generalization.
- **Accuracy**: `0.2134` over 100 playlists.
- **Remarks**: 
  - The reduction in training data led to a drop in accuracy, as expected (~21.3%). 
  - The model might benefit from a better train-test split, particularly ensuring that the data is shuffled to remove any bias towards specific sections of the dataset.

### General Observations and Future Steps
- **Need for Larger Test Sets**: All models should be tested against a much larger set of playlists, as testing on only 100 playlists may not provide sufficient variability and insight into model performance.
- **Train-Test Split**: Proper partitioning between training and testing datasets is essential to prevent overfitting and ensure the model isn’t just learning patterns specific to the training data.
- **Accuracy Potential**: It remains to be seen how high the accuracy can reach with these approaches, but further experimentation and optimization are necessary.

### Next Steps
- Expand testing to larger test sets for more reliable evaluation.
- Shuffle the data before splitting into training and testing to reduce bias and improve generalization.
- Continue optimizing models to explore the upper limits of achievable accuracy with this framework.



## learning algorithms
### CBOW (Continuous-bag-of-words)

### Skip-Gram

### Conclusion / work in progress
CBOW is better for Datasets with many fequent occuring words. Skip-Gram is better with many less frequent words ([Concluded here](https://iopscience.iop.org/article/10.1088/1742-6596/2634/1/012052/meta)). Still need to test what algorithm performes better or is it the case to create an own algorithm for embedding songs inside playlists.  Both algorithms work with some kind of context window to understand the focused word. This type of window looking should not apply to playlists because there is the whole list relevant.

## ideas
could also take a dataset of playlist
with songs in it and embedding them like the "sentences" to get the style of the individual songs. Would be one step in the direction of understanding the songs.

CBOW will be the main focuse here at 

Cleaning up the dataset for outliers to polish the accuracy of the trained model

It's possible that I need to implement my own model for embedding those words of the playlist to let the context window slip.

Following step would be to get the embedding algorithms working with mp3 data.


## Relevant Papers
Zarlenga, M.E., Barbiero, P., Ciravegna, G., Marra, G., Giannini, F., Diligenti, M., Precioso, F., Melacci, S., Weller, A., Lio, P. and Jamnik, M., 2022, November. [Concept embedding models](https://hal.science/hal-03854550/). In NeurIPS 2022-36th Conference on Neural Information Processing Systems.

Xia, H., 2023, November. [Continuous-bag-of-words and Skip-gram for word vector training and text classification](https://iopscience.iop.org/article/10.1088/1742-6596/2634/1/012052/meta). In Journal of Physics: Conference Series (Vol. 2634, No. 1, p. 012052). IOP Publishing.
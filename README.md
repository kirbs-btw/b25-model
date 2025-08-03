# b25 - Understanding song relations
## Overview
This project explores how **word embedding techniques**—commonly used in **natural language processing (NLP)**—can be adapted for **music recommendations**. Traditional NLP algorithms like **CBOW (Continuous-Bag-of-Words)**, **Skip-Gram**, and **GloVe** typically rely on the idea that words are related when they appear near each other in text. In contrast, many recommendation scenarios (such as playlists) don’t depend on strict ordering, but rather on **co-occurrence within sets**. This work demonstrates how to modify these algorithms for **non-sequential data** (e.g., songs that appear together in playlists).


## Key Contributions
1. **Adapting Established Models**  
   - **CBOW** and **Skip-Gram**: Repurposed for recommendation by treating each playlist as a “bag of items” rather than a sequence of words.
   - **GloVe-Inspired Approach**: Harnesses global co-occurrence statistics of songs, without relying on their relative positions in a playlist.

2. **Introducing Novel Algorithms**  
   - **CBOE (Continuous-Bag-of-Entities)**: A generalization of CBOW that removes explicit context windows, making it better suited for non-sequential data.  
   - **SGE (Skip-Gram for Entities)**: Adapts Skip-Gram to capture relationships in set-like data, allowing an “infinite” context window and equal weighting for co-occurring entities.  
   - **ECP (Entity Cluster Push)**: A simplified, cluster-oriented method inspired by CBOW. Instead of predicting a target entity, ECP focuses on pushing entity vectors closer together within each cluster (playlist).

## Methodology
1. **Data Preprocessing**  
   - Curated Spotify playlists serve as the dataset.  
   - Randomly rearranged playlists for robust training.  
   - 90/10 train-test split ensures reliable evaluation.

2. **Training**  
   - **CBOW** & **Skip-Gram**: Adapted for set-based contexts.  
   - **GloVe-Inspired**: Builds a global co-occurrence matrix for playlists.  
   - **CBOE, SGE, ECP**: Original algorithms designed for entity (song) embeddings in non-sequential data.

3. **Evaluation**  
   - **F1 Score** used as the primary metric.  
   - **Precision**: Fraction of recommended songs that actually belong in the playlist.  
   - **Recall**: Fraction of actual songs correctly recommended by the model.

## Contributing
We welcome contributions via pull requests. For major changes, please open an issue first to discuss potential improvements.

## References
This work builds on the principles of distributional semantics and recommendation systems, referencing studies like:
- Pilehvar & Camacho-Collados (2020) on **word embeddings**  
- Sahlgren (2008) on the **distributional hypothesis**  
- Wang et al. (2017, 2020) on vector-based recommendation methods  
- Musto et al. (2015) on applying **word embeddings** to recommender systems  
- And others noted throughout the documentation


## notes 
data quality is bad 
need for adjustmens in the min word count 

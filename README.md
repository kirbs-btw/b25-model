# b25 – Embedding Songs in Playlist Contexts

> Exploring how NLP-inspired word embedding models can be adapted for music recommendation by learning song relations from playlist co-occurrence data.

---

## Abstract

This repository contains **b25**, an experimental framework that adapts classical word embedding techniques from natural language processing (NLP) to **playlist-based music recommendation**.

By treating playlists as analogues of sentences and songs as entities (words), the project investigates how models such as **CBOW**, **Skip-Gram**, and a **GloVe-inspired global co-occurrence model**, as well as entity-centric variants **CBOE**, **SGE**, and a cluster-based **ECP** update rule, can learn meaningful song embeddings from **weakly ordered / set-like** playlist data.

Experiments show that models which exploit **global co-occurrence structure**, in particular larger Skip-Gram variants with broad contexts, achieve strong recommendation performance (up to about **77.39% precision@1** in internal experiments), while the introduced **Entity Cluster Push (ECP)** prototype underperforms and requires further refinement.

---

## 1. Motivation

Classical word embeddings rely on the **distributional hypothesis**:

> Words that occur in similar contexts tend to have similar meanings.

In this project, we transfer that idea from text to music:

- **Words** → songs  
- **Sentences** → playlists  
- **Corpus** → collection of curated Spotify playlists  

Unlike sentences, playlists are often **weakly ordered**: the semantic relation between two songs is more about **co-occurrence in the same playlist** than about their exact adjacency. This motivates embedding models that focus on **sets / clusters of entities** instead of strict sequences.

---

## 2. Problem Formulation

Let

- $\mathcal{V}$ be the set of all songs (the vocabulary),
- a playlist be a finite subset (or multiset) $P \subseteq \mathcal{V}$,
- a dataset of playlists be $\mathcal{P} = \{ P_1, \dots, P_M \}$.

We seek an embedding function

$$
f : \mathcal{V} \to \mathbb{R}^d, \quad e \mapsto \mathbf{v}_e,
$$

such that **co-occurring songs in playlists** have **similar vectors** in $\mathbb{R}^d$.

### 2.1 Co-occurrence Matrix

For global methods we define a symmetric co-occurrence matrix $X \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ by

$$
X_{ij} = \text{number of playlists in which songs } e_i \text{ and } e_j \text{ co-occur}.
$$

In practice, we are interested in pairs with $X_{ij} > 0$ (i.e., songs that co-occur at least once). This matrix is the basis for the GloVe-inspired model and informs the design of the entity-based variants.

---

## 3. Models Implemented

### 3.1 Baseline NLP-Inspired Models

#### 3.1.1 CBOW (Continuous-Bag-of-Words)

CBOW predicts a **center word** $w_t$ from its **context words** within a window around it. Let $\text{context}(w_t)$ denote the multiset of words in that window. The classical objective is

$$
\max_{\theta} \sum_{t=1}^{T} 
\log P\big(w_t \mid \text{context}(w_t); \theta\big),
$$

where $P(\cdot \mid \cdot)$ is typically parameterized using the average of the embeddings of the context words and a softmax over the vocabulary. The model learns distributed representations such that words appearing in similar contexts have similar embeddings.

*(In b25, CBOW is used as a conceptual baseline; the playlist-specific adaptation is formalized as CBOE in Section 3.2.1.)*

---

#### 3.1.2 Skip-Gram (SG)

Skip-Gram inverts CBOW: given a **center word** $w_t$, it predicts each surrounding context word $w_{t+j}$ in a fixed window. The classic objective is

$$
\max_{\theta} \sum_{t=1}^{T}
\sum_{\substack{-c \le j \le c \\ j \ne 0}}
\log P(w_{t+j} \mid w_t; \theta),
$$

where $c$ is the context window size.

The conditional probability is typically modelled using a softmax:

$$
P(w_o \mid w_I) =
\frac{\exp(\mathbf{v}_{w_o}^\top \mathbf{v}_{w_I})}
{\sum_{w=1}^W \exp(\mathbf{v}_w^\top \mathbf{v}_{w_I})},
$$

with $\mathbf{v}_{w_I}$ and $\mathbf{v}_{w_o}$ being input and output embeddings.

*(In b25, the playlist-specific Skip-Gram variant is formalized as SGE in Section 3.2.2.)*

---

#### 3.1.3 GloVe-Inspired Global Co-Occurrence Embedding

Following GloVe, we use the global co-occurrence matrix $X$ and optimize embeddings $\mathbf{w}_i, \tilde{\mathbf{w}}_j \in \mathbb{R}^d$ and biases $b_i, \tilde{b}_j \in \mathbb{R}$ via a **weighted least-squares loss**:

$$
J_{\text{GloVe}} =
\sum_{\substack{i,j \\ X_{ij} > 0}} f(X_{ij})
\left(
\mathbf{w}_i^\top \tilde{\mathbf{w}}_j
+ b_i + \tilde{b}_j
- \log X_{ij}
\right)^2,
$$

where $f(\cdot)$ is a weighting function that down-weights very rare or extremely frequent co-occurrences. Restricting the sum to $X_{ij} > 0$ ensures that $\log X_{ij}$ is well-defined.

In b25, co-occurrence counts $X_{ij}$ come from **playlist-level co-inclusion** rather than local windows in a text.

---

### 3.2 Entity-Centric Playlist Models

The following models are designed specifically for **non-sequential / weakly ordered entity sets** like playlists, where **co-occurrence within a set** is more important than position.

#### 3.2.1 CBOE – Continuous-Bag-of-Entities

CBOE adapts CBOW to sets of entities by:

- removing the notion of a **fixed context window**, and  
- treating the entire playlist (minus the target entity) as the context.

For a playlist $P = \{ e_1, \dots, e_n \}$ and target entity $e_k$, we define the **context vector**

$$
\mathbf{c}_{e_k} =
\frac{1}{|P| - 1}
\sum_{e \in P \setminus \{e_k\}}
\mathbf{v}_e.
$$

The model then maximizes the likelihood of $e_k$ given $\mathbf{c}_{e_k}$, analogous to CBOW but geared towards **unordered entity sets**:

$$
\max_{\theta} \sum_{P \in \mathcal{P}}
\sum_{e_k \in P}
\log P(e_k \mid \mathbf{c}_{e_k}; \theta).
$$

In other words, **CBOE is CBOW applied to playlists treated as bags of entities**.

---

#### 3.2.2 SGE – Skip-Gram for Entities

SGE adapts Skip-Gram to entity sets where **all co-occurring entities are treated symmetrically**. Instead of a positional window, every pair of distinct songs in the same playlist forms a training signal. For each playlist $P$,

$$
J_{\text{SGE}} =
\sum_{P \in \mathcal{P}}
\sum_{e_i \in P}
\sum_{\substack{e_j \in P \\ j \ne i}}
\log P(e_j \mid e_i; \theta),
$$

with the same softmax parameterization as standard Skip-Gram, but **no positional window or distance-based weighting**. This corresponds to an “infinite” window over the set $P$.

---

#### 3.2.3 ECP – Entity Cluster Push

Entity Cluster Push (ECP) is a cluster-oriented embedding update rule that directly **pulls entity vectors towards their cluster centroid**.

For a playlist (cluster) $P = \{ e_1, \dots, e_n \}$, we define the **base vector**

$$
\mathbf{b} = \sum_{i=1}^n \mathbf{v}_{e_i}.
$$

For a given entity $e_k \in P$, its **context (centroid) vector** is

$$
\mathbf{c}_{e_k} =
\frac{\mathbf{b} - \mathbf{v}_{e_k}}{n - 1},
$$

i.e., the mean embedding of all other entities in the cluster.

We can measure their Euclidean distance as

$$
\text{dist}_k = \left\| \mathbf{v}_{e_k} - \mathbf{c}_{e_k} \right\|_2,
$$

and update the entity embedding via a step **towards** this centroid:

$$
\mathbf{v}_{e_k}
\leftarrow
\mathbf{v}_{e_k} + \alpha \big( \mathbf{c}_{e_k} - \mathbf{v}_{e_k} \big),
$$

where $\alpha$ is a learning rate. This update decreases the distance between $\mathbf{v}_{e_k}$ and $\mathbf{c}_{e_k}$ (for $0 < \alpha < 1$), thereby pulling entities in a playlist closer together in embedding space.

The base vector $\mathbf{b}$ is updated accordingly to reflect changes in $\mathbf{v}_{e_k}$, and this process is iterated over all entities in all clusters. Unlike CBOW/Skip-Gram, **ECP does not predict a target**; it directly optimizes the **relative geometry** of entity embeddings within playlists.

---

## 4. Data & Preprocessing

- **Data source:** curated Spotify playlists (not included in this repository due to licensing terms).
- **Cleaning:** removal of obviously erroneous entries, handling of missing or malformed data.
- **Augmentation:** playlists are randomly **re-shuffled** to reduce any artefacts from the original order and encourage models to focus on co-occurrence rather than sequence.
- **Split:** a **90% / 10% train–test split** over playlists is used.

Data quality and coverage (especially for long-tail tracks) are key limitations and are discussed in more detail in the accompanying paper draft.

---

## 5. Evaluation Protocol

For evaluation, each playlist is treated as a separate test case.

Given a **seed song** $s$ from a playlist $P$:

1. The model retrieves candidate songs $\hat{P}(s)$ based on embedding similarity.
2. A recommendation is considered **correct** if the predicted song appears in the same playlist $P$.

### 5.1 Precision, Recall, and F1

Let

- $TP$ = number of correctly recommended songs,  
- $FP$ = number of recommended songs not in the playlist,  
- $FN$ = number of relevant songs in the playlist that were not recommended.

Then

$$
\text{Precision} = P = \frac{TP}{TP + FP},
$$

$$
\text{Recall} = R = \frac{TP}{TP + FN},
$$

$$
F_1 = 2 \cdot \frac{P \cdot R}{P + R}.
$$

We also report **precision@k**, e.g., precision@1 for the top-ranked recommendation.

---

## 6. Repository Structure

High-level layout:

- `ecp/` – implementation and experiments for **Entity Cluster Push (ECP)**.
- `sge/` – Skip-Gram for Entities (**SGE**) models and experiments.
- `entity2vec/` – shared embedding utilities and common components.
- `notebooks/` – exploratory notebooks (model training, embedding visualization, analysis).
- `scripts/` – helper scripts for data processing and experiment orchestration.
- `data/` – local placeholder for playlist data (not committed in full due to licensing / size).
- `tests/` – minimal tests for core components.

See individual subdirectories for model-specific details and experiment configurations.

---

## 7. Key Findings (Short Summary)

- **Global co-occurrence–aware models** (e.g., wide-context, playlist-level Skip-Gram variants like SGE) perform best in this setup, achieving up to **≈77.39% precision@1** in internal experiments.
- Purely proximity-based word-embedding strategies that rely heavily on sequential order are less well suited for playlist data.
- **ECP**, while conceptually appealing for clusters, **underperformed** baselines in its current form and requires further tuning and potentially more expressive update rules.

---

## 8. Limitations & Future Work

**Limitations**

- **Data coverage:** sparse occurrence of many songs leads to unreliable embeddings in the long tail.
- **Subjectivity:** music preference is inherently subjective; playlist-level evaluation cannot fully capture user satisfaction.
- **Hardware constraints:** limited compute resources constrained embedding dimensionality and hyperparameter sweeps.

**Future directions**

- More systematic **hyperparameter studies** (embedding size, negative sampling, learning rate, etc.).
- Integration of **content features** (audio, metadata, lyrics) to complement co-occurrence-based embeddings.
- Scaling to larger, more diverse playlist corpora and additional domains (movies, podcasts, retail items).
- Refinement of **ECP** and other cluster-based embedding updates; exploration of hybrid models that mix predictive and geometric objectives.

---

## 9. How to Use This Repository

1. Create a Python environment and install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Place your playlist data in the appropriate data/ directory structure (see notebooks for expected format).

3. Use the notebooks and scripts to:
- build the co-occurrence matrix,
- train different embedding variants (CBOW, Skip-Gram, GloVe-like, CBOE, SGE, ECP),
- evaluate precision, recall, F1, and precision@k.

## 10. Citation
If you reference this work in academic or research settings, you can cite it as:
> Bastian Lipka. b25: Embedding Songs in Playlist Contexts – Adapting Word Embedding Techniques for Music Recommendation Systems, 2025. Manuscript in preparation.

A BibTeX-style stub:
```bibtex
@unpublished{lipka2025b25,
  author  = {Lipka, Bastian},
  title   = {b25: Embedding Songs in Playlist Contexts -- Adapting Word Embedding Techniques for Music Recommendation Systems},
  year    = {2025},
  note    = {Manuscript in preparation},
}
```

## 11. References
Lasmedi Afuan et al. “Implementations of The Fuzzy C-Means Clustering Method in Song List Based on Emotional Mood.” ICICoS 2024.

Hsiao-Yu Chiang, Jose Camacho-Collados, and Zachary Pardos. “Understanding the Source of Semantic Regularities in Word Embeddings.” CoNLL 2020.

Casper Hansen et al. “Contextual and Sequential User Embeddings for Large-Scale Music Recommendation.” RecSys 2020.

Burak Köse, Süleyman Eken, and Ahmet Sayar. “Playlist Generation via Vector Representation of Songs.” In Advances in Big Data, 2016.

Ana Laura Lezama-Sánchez, Mireya Tovar Vidal, and José A. Reyes-Ortiz. “An Approach Based on Semantic Relationship Embeddings for Text Classification.” Mathematics, 2022.

Mohammad Taher Pilehvar and Jose Camacho-Collados. Embeddings in Natural Language Processing: Theory and Advances in Vector Representations of Meaning. Morgan & Claypool, 2020.

Magnus Sahlgren. “The Distributional Hypothesis.” Italian Journal of Linguistics, 2008.

S. Selva Birunda and R. Kanniga Devi. “A Review on Word Embedding Techniques for Text Classification.” In Innovative Data Communication Technologies and Application, 2021.

Camila Sundermann et al. “Exploration of Word Embedding Model to Improve Context-Aware Recommender Systems.” IEEE/WIC/ACM Web Intelligence, 2018.

Dongjing Wang et al. “CAME: Content- and Context-Aware Music Embedding for Recommendation.” IEEE Transactions on Neural Networks and Learning Systems, 2020.

Qi Wang et al. “Two Improved Continuous Bag-of-Word Models.” IJCNN, 2017.

Yahui Wang and Paul Craig. “Vector Space Model Embedding for Recommender System Neural Networks.” ICSAI, 2017.

Haowen Xia. “Continuous-Bag-of-Words and Skip-Gram for Word Vector Training and Text Classification.” Journal of Physics: Conference Series, 2023.

Cataldo Musto et al. “Word Embedding Techniques for Content-Based Recommender Systems: An Empirical Evaluation.” RecSys Posters, 2015.
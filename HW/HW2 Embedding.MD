## 1 Word2Vec
### 1.1 Preprocess the text data

- Remove newline ('\n')
- Remove carriage returns ('\r')
- Remove punctuations
- Remove numbers
- Convert to lower-case
- Split each sentence into a list of words

### 1.2 Train a Word2Vec model
#### We will be using Word2Vec
#### Word2Vec converts words into vectors. Words that appear in similar contexts tend to have similar embeddings (distributional hypothesis).

## 2 Dimensionality Reduction
### High-dimensional data can't be visualized directly, so we need to compress it down to 2D while trying to preserve the relationships between points.
### 2.1 t-SNE (t-distributed Stochastic Neighbor Embedding)
```python
tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
Y_subset_tsne = tsne.fit_transform(X_subset)
```

### 2.2 UMAP (Uniform Manifold Approximation and Projection)
```python
reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
Y_subset_umap = reducer.fit_transform(X_subset)
```

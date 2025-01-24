# %% Import
import requests
import numpy as np
import pandas as pd
from scipy import sparse
import umap
from sklearn.decomposition import PCA
import spacy
from tqdm.auto import tqdm
import gensim
import matplotlib.pyplot as plt
import seaborn as sns

# %% Test -----------
import requests

# Download Les Misérables text from Project Gutenberg
emb = np.random.randn(100, 32)
word_to_idx = {f"word_{i}": i for i in range(100)}
kv = gensim.models.KeyedVectors(emb.shape[1])
kv.add_vectors([word for word in word_to_idx.keys()], emb)

# If the function is implemented correctly, the following test should pass.
for i in range(1000):
    sampled = np.random.choice(kv.index_to_key, size=2, replace=False)
    pos_word, neg_word = sampled[0], sampled[1]
    pos_vec = kv.get_vector(pos_word).copy()
    neg_vec = kv.get_vector(neg_word).copy()
    _semaxis_scores = get_semaxis(np.array([[pos_vec - neg_vec],[neg_vec - pos_vec]]).reshape(2,-1), kv, pos_word, neg_word)

    assert np.allclose(_semaxis_scores[0], 1, atol=1e-2), "The semaxis scores are not correctly computed"
    assert np.allclose(_semaxis_scores[1], -1, atol=1e-2), "The semaxis scores are not correctly computed"

# %% Import
import requests
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import gensim
import matplotlib.pyplot as plt
import seaborn as sns


# %% Test -----------

tfidf_matrix = np.random.randn(100, 100)

emb = embedding_characters(tfidf_matrix)

# If you could compute the embedding correctly, the following test should pass.
assert emb.shape[0] == tfidf_matrix.shape[0], "The number of words is not correct"
assert emb.shape[1] == 32, "The embedding dimension is not 32"
assert np.mean(np.linalg.norm(emb, axis=0)) > 0.99, "The embedding is not normalized"
assert np.allclose(np.mean(emb, axis=0), np.zeros(32), atol=1e-2), "The embedding is not centered"



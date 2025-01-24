# %% Import
import requests
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


# %% Test -----------

# Download Les Misérables text from Project Gutenberg
url = "https://www.gutenberg.org/cache/epub/135/pg135.txt"
response = requests.get(url)
text = response.text[1000: 100000].lower()
chunk_size = 1000

docs = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

tfidf_matrix, word_to_idx = calculate_tf_idf_matrix(docs)

data = np.load("data/tfidf_matrix_signature.npz")
s_ref = data["s"]

_, s, _ = sparse.linalg.svds(tfidf_matrix, k=10)

assert np.allclose(np.argsort(s),np.argsort(s_ref), atol=1e-3), "The tfidf matrix is not correctly computed."

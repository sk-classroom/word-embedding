

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
url = "https://www.gutenberg.org/cache/epub/135/pg135.txt"
response = requests.get(url)
text = response.text.lower()[500: 50000]

chunk_size = 1000
docs = [text[i: i+chunk_size] for i in range(0, len(text), chunk_size)]
emb, word_to_idx = generate_word2vec_embeddings(docs)

df_characters = pd.DataFrame([
    ['myriel', '', 'male'],
    ['napoleon', '', 'male'],
    ['cravatte', '', 'male'],
    ['labarre', '', 'male'],
    ['jean', 'valjean_household', 'male'],
    ['marguerite', '', 'female'],
    ['isabeau', '', 'female'],
    ['gervais', '', 'male'],
    ['listolier', 'young_paris_circle', 'male'],
    ['fameuil', 'young_paris_circle', 'male'],
    ['favourite', 'young_paris_circle', 'female'],
    ['dahlia', 'young_paris_circle', 'female'],
    ['fantine', 'young_paris_circle', 'female'],
    ['cosette', 'valjean_household', 'female'],
    ['javert', 'trial_characters', 'male'],
    ['fauchelevent', 'valjean_household', 'male'],
    ['bamatabois', '', 'male'],
    ['simplice', 'trial_characters', 'female'],
    ['scaufflaire', 'trial_characters', 'male'],
    ['judge', 'trial_characters', 'male'],
    ['champmathieu', 'trial_characters', 'male'],
    ['brevet', 'trial_characters', 'male'],
    ['chenildieu', 'trial_characters', 'male'],
    ['cochepaille', 'trial_characters', 'male'],
    ['pontmercy', '', 'male'],
    ['boulatruelle', '', 'male'],
    ['gribier', '', 'male'],
    ['jondrette', '', 'male'],
    ['gavroche', '', 'male'],
    ['gillenormand', '', 'male'],
    ['magnon', '', 'female'],
    ['marius', '', 'male'],
    ['mabeuf', '', 'male'],
    ['enjolras', 'revolutionary_core', 'male'],
    ['combeferre', 'revolutionary_core', 'male'],
    ['prouvaire', 'revolutionary_core', 'male'],
    ['feuilly', 'revolutionary_core', 'male'],
    ['courfeyrac', 'revolutionary_core', 'male'],
    ['bahorel', 'revolutionary_core', 'male'],
    ['bossuet', 'revolutionary_core', 'male'],
    ['joly', 'revolutionary_core', 'male'],
    ['grantaire', 'revolutionary_core', 'male'],
    ['gueulemer', 'thenardier_gang', 'male'],
    ['babet', 'thenardier_gang', 'male'],
    ['claquesous', 'thenardier_gang', 'male'],
    ['montparnasse', 'thenardier_gang', 'male'],
    ['toussaint', 'valjean_household', 'female'],
    ['brujon', 'thenardier_gang', 'male']
], columns=['character', 'group', 'gender'])

# Remove the characters that are not in the word_to_idx
df_characters = df_characters[df_characters['character'].isin(word_to_idx.keys())]

assert emb.shape[1] == 32, "The embedding dimension is not 32"
assert emb.shape[0] == len(word_to_idx), "The number of words is not correct"

les_miserables_characters = df_characters['character'].tolist()
emb_characters  = np.zeros((len(les_miserables_characters), emb.shape[1]))
for i, character in enumerate(les_miserables_characters):
    emb_characters[i, :] = emb[word_to_idx[character], :]

focal_groups = ['revolutionary_core', 'young_paris_circle', "thenardier_gang"]
n_emb_characters = np.einsum('ij,i->ij', emb_characters, 1.0 / np.linalg.norm(emb_characters, axis=1))
group_centroids = np.array([np.mean(n_emb_characters[df_characters['group'] == group], axis=0) for group in focal_groups])
group_similarieis = group_centroids @ group_centroids.T
within_group_similarities = np.mean(np.diag(group_similarieis))
between_group_similarities = np.sum(group_similarieis - np.diag(group_similarieis)) / (len(focal_groups) * (len(focal_groups) - 1))

assert within_group_similarities > between_group_similarities, "The groups are not well separated"


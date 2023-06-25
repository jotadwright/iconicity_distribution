import pandas as pd
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import panphon
import epitran

# Assign Panphon and Epitran objects.
ft = panphon.FeatureTable()
epi = epitran.Epitran('eng-Latn')


def vec_norm(vec_list):
    """Function reassigns -1s as 0s in a vector."""
    new_list = []
    for v in vec_list:
        v_n = [0 if item == -1 else item for item in v]
        new_list.append(v_n)
    return new_list


def sem_vec(word):
    """Function gets semantic embedding for each word and gives an empty vector for out-of-vocabulary words"""
    vec_len = len(ds_model.get_vector('cat'))

    if ds_model.has_index_for(word):
        vec = ds_model.get_vector(word)
    else:
        vec = [0] * vec_len

    return vec


def rescale(xmin, xmax, ymin, ymax, data):
    """Function rescales from range x to range y"""
    data_scaled = np.interp(data, (xmin, xmax), (ymin, ymax))
    return data_scaled

# Choose new dataset (Winter et al., 2023) or old dataset (Winter et al., 2017)
dataset_new = True

# Open iconicity dataset, rescale iconicity values and sort data.
if not dataset_new:
    df = pd.read_csv('data/iconicity_ratings.csv')
    df['Iconicity'] = df['Iconicity'].apply(lambda x: rescale(-5, 5, -1, 1, x))
    df = df.sort_values(by='Iconicity', ascending=False, ignore_index=True)

if dataset_new:
    df = pd.read_csv('data/iconicity_ratings_cleaned.csv')
    df = df[['word', 'rating']]
    df = df.rename(columns={"word": "Word", "rating": "Iconicity"})
    df['Word'] = df['Word'].astype(str)
    df['Iconicity'] = df['Iconicity'].apply(lambda x: rescale(1, 7, 0, 1, x))
    df = df.sort_values(by='Iconicity', ascending=False, ignore_index=True)

# Get IPA transliteration of all words.
df['ipa'] = df['Word'].apply(epi.transliterate)

# Get Panphon vectors and normalise.
df['phon_vec'] = df['ipa'].apply(lambda x: ft.word_to_vector_list(x, numeric=True))
df['phon_vec'] = df['phon_vec'].apply(vec_norm)

# Get Fasttext vectors. Pretrained vectors available at https://fasttext.cc/docs/en/english-vectors.html
print("Loading DS model...")
ds_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)
print("DS model loaded.")
words_covered = list(ds_model.index_to_key)

df['sem_vec'] = df['Word'].apply(sem_vec)

# Save to pickle.
df.to_pickle('data/iconicity_vectors.pkl')

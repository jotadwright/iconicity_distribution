# iconicity_embeddings
Research on the presence of iconic information in phonetic and semantic word embeddings.

Uses Winter et al. (2023) iconicity ratings (https://github.com/bodowinter/iconicity_ratings), Fasttext semantic embeddings (https://fasttext.cc/docs/en/english-vectors.html) and Panphon phonetic embeddings (https://github.com/dmort27/panphon)

iconicity_vectors.py gets semantic and phonetic embeddings for the dataset.
phon_corr.py gives analysis of the correlation between phonetic features.
PCA_search.py gives analysis of best performing semantic model dimensionality.
phonetic_model.py, semantic_model.py, and combined_model.py run the three main models used in the research.
analysis.py gives comparative analysis of the models

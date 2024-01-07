# The Semantic and Phonetic Distribution of Iconicity in English: A Computational Study

Iconicity is a perceived similarity between aspects of the form of a word or sign and its meaning. This study makes use of iconicity ratings for English, derived from native speaker judgements, and embeddings based on distributional semantic and articulatory phonetic information to model the relationships between iconicity, semantics and phonetics for a vocabulary of over 14,000 English words. Linear regression models are employed to connect these elements, investigating the extent to which iconicity information is captured in English phonetic and semantic word embeddings. The models successfully predict iconicity ratings to a certain degree, suggesting the presence of iconicity information in these embeddings. Moreover, the models' predictions align with patterns found in previous research on the phonetic and semantic dimensions of iconicity. This research shines light on the relationship between iconicity, form and meaning, and contributes to our knowledge of the cognitive and linguistic mechanisms underlying the phenomenon of iconicity.

Uses Winter et al. (2023) iconicity ratings (https://github.com/bodowinter/iconicity_ratings), Fasttext semantic embeddings (https://fasttext.cc/docs/en/english-vectors.html) and Panphon phonetic embeddings (https://github.com/dmort27/panphon)

iconicity_vectors.py gets semantic and phonetic embeddings for the dataset.

phon_corr.py gives analysis of the correlation between phonetic features.

PCA_search.py gives analysis of best performing semantic model dimensionality.

phonetic_model.py, semantic_model.py, and combined_model.py run the three main models used in the research.

analysis.py gives comparative analysis of the models

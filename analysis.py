import pandas as pd
import numpy as np
from scipy.stats import kendalltau


def rescale(xmin, xmax, ymin, ymax, data):
    """Function rescales from range x to range y"""
    data_scaled = np.interp(data, (xmin, xmax),(ymin, ymax))
    return data_scaled


def kendcorr(X, Y):
    """Function gets scipy Kendall's correlation between lists X and Y"""
    corr, p = kendalltau(X, Y)

    return corr, p

# Get predictions from the three models and combine into one dataframe.
phon = pd.read_csv("data/icon_phon.csv")
sem = pd.read_csv("data/icon_sem.csv")
comb_pred = pd.read_csv("data/comb_pred.csv")
sem = sem.drop(['Word', 'Iconicity'], axis=1)
df = pd.concat([phon, sem], axis=1, join='inner')
df['comb_pred'] = comb_pred
df = df[['Word', 'ipa', 'Iconicity', 'phon_pred', 'sem_pred', 'comb_pred']]

# Get difference in predictions between phonetic and semantic models.
df['pred_diff'] = df['phon_pred'] - df['sem_pred']

# Get concreteness ratings for dataset and rescale
conc = pd.read_csv('data/concreteness.csv')
conc = conc[["Word","Conc.M"]]
df = pd.merge(df, conc, on="Word", how="left")
df['Conc.M'] = df['Conc.M'].apply(lambda x: rescale(1, 5, 0, 1, x))

# Get rank correlation between model predictions and iconicity.
phon_tau = kendcorr(df['Iconicity'],df['phon_pred'])
sem_tau = kendcorr(df['Iconicity'],df['sem_pred'])
comb_tau = kendcorr(df['Iconicity'],df['comb_pred'])

# Get rank correlation between model predictions and iconicity.
sem_conc_tau = kendcorr(df['Conc.M'],df['sem_pred'])

print(f'Phonetic prediction tau: {phon_tau}')
print(f'Semantic prediction tau: {sem_tau}')
print(f'Combined prediction tau: {comb_tau}')

# Sort by top predictions.
phon_max = df.sort_values(by='phon_pred', ascending=False, ignore_index=True).round(3)
sem_max = df.sort_values(by='sem_pred', ascending=False, ignore_index=True).round(3)
comb_max = df.sort_values(by='comb_pred', ascending=False, ignore_index=True).round(3)

# Save each model's top predictions to separate CSV.
phon_max.to_csv('data/phon_max.csv', index=False)
sem_max.to_csv('data/sem_max.csv', index=False)
comb_max.to_csv('data/comb_max.csv', index=False)





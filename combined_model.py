import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics

# Open datasets for phonetic and semantic models and combine.
phon = pd.read_csv("icon_phon.csv")
sem = pd.read_csv("icon_sem.csv")
sem = sem.drop(['Word', 'Iconicity', 'sem_pred'], axis=1)
df = pd.concat([phon, sem], axis=1, join='inner')

# Save to CSV.
df.to_csv('data/icon_combined.csv')

# Remove unnecessary phonetic features.
remove = ['10_sg', '11_cg', '20_velaric', '22_long', '23_hitone', '24_hireg']
df = df.drop(remove, axis=1)

X = df.drop(['Word', 'Iconicity', 'ipa', 'phon_pred'], axis=1)
y = df['Iconicity']

# Load and fit the linear regression model to the training set.
lm = linear_model.LinearRegression()
lm.fit(X, y)

# Get 5-fold cross-validation scores for the training set
folds = KFold(n_splits=5, shuffle=True, random_state=33)
scores = cross_val_score(lm, X, y, scoring='r2', cv=folds)

print('K-fold CV R squared:', scores, "\n")

# Get predictions and print a sample
y_pred = lm.predict(X)

print("Sample of predictions")
lm_diff = pd.DataFrame({'iconicity': y, 'lm_pred': y_pred})
lm_diff = lm_diff.join(df['Word']).sort_index()
print(lm_diff.sample(20).sort_values(by='iconicity', ascending=False, ignore_index=True), "\n")

# Get and print LM test set metrics and coefficients
meanAbErr = metrics.mean_absolute_error(y, y_pred)
meanSqErr = metrics.mean_squared_error(y, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y, y_pred))

print(f'Coefficients {lm.coef_}')

print('\n', 'Test set metrics')
print('R squared: {:.2f}'.format(lm.score(X, y) * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

df['comb_pred'] = y_pred

df['comb_pred'].to_csv('data/comb_pred.csv', index=False)

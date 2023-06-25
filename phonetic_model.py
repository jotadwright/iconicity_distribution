import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics

# Get dataset and open in dataframe.
df = pd.read_csv('data/icon_phon.csv')

# Assign names for all phonetic features.
phon_features = ['1_syl', '2_son', '3_cons', '4_cont', '5_delrel', '6_lat', '7_nas', '8_strid', '9_voi', '10_sg',
                 '11_cg', '12_ant',
                 '13_cor', '14_distr', '15_lab', '16_hi', '17_lo', '18_back', '19_round', '20_velaric', '21_tense',
                 '22_long', '23_hitone', '24_hireg']

# Assign names for phonetic features to keep.
keep = ['1_syl', '2_son', '3_cons', '4_cont', '5_delrel', '6_lat', '7_nas', '8_strid',
        '9_voi', '12_ant', '13_cor', '14_distr', '15_lab', '16_hi', '17_lo', '18_back', '19_round', '21_tense']

# Make linear regression model with phonetic features.
X = df[keep]
y = df['Iconicity']

lm = linear_model.LinearRegression()
reg = lm.fit(X, y)

y_pred = lm.predict(X)

# Get performance metrics.
meanAbErr = metrics.mean_absolute_error(y, y_pred)
meanSqErr = metrics.mean_squared_error(y, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y, y_pred))

# Print coefficients
for i, j in zip(keep, lm.coef_):
    print(i, j)

# Print performance metrics
print('R squared: {:.2f}'.format(lm.score(X, y) * 100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)

# Save model predictions to dataframe.
df['phon_pred'] = y_pred

# Save to CSV.
df = df.sort_values(by='phon_pred', ascending=False, ignore_index=True)
df.to_csv('data/icon_phon.csv', index=False)

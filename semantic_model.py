import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import metrics


def reduce_dim(vector, n):
    """Reduces dimensionality of a vector to n via PCA"""
    pca = PCA(n_components=n)
    nd_vec = pca.fit_transform(vector)
    return nd_vec


def lin_reg(X, y):
    """Function runs sklearn linear regression model and returns predictions"""
    # Load and fit the linear regression model to the training set.
    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    # Get predictions and print a sample
    y_pred = lm.predict(X)

    # Get and print LM test set metrics
    meanAbErr = metrics.mean_absolute_error(y, y_pred)
    meanSqErr = metrics.mean_squared_error(y, y_pred)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y, y_pred))

    print('Test set metrics')
    print('R squared: {:.2f}'.format(lm.score(X, y) * 100))
    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)

    print('Intercept:', lm.intercept_)

    coef = lm.coef_

    print(coef)

    return y_pred


# Load iconicity dataset
df = pd.read_pickle('data/iconicity_vectors.pkl')

# Set number of dimensions for PCA and reduce dimensionality
dimensions = 185
sem_vec_red = reduce_dim(df['sem_vec'].to_list(), dimensions)

# Create numbered dimension columns
dim_col = []
for d in range(0, dimensions):
    name = f'S{d + 1}'
    dim_col.append(name)

# Split dimensions into separate columns
df['sem_vec_red'] = sem_vec_red.tolist()
spl = pd.DataFrame(df['sem_vec_red'].to_list(), columns=dim_col)
df = df.join(spl)
df = df.drop(['sem_vec', 'sem_vec_red', 'ipa', 'phon_vec'], axis=1)

# Run linear regression and save predictions
df['sem_pred'] = lin_reg(sem_vec_red, df['Iconicity'])
df.to_csv('data/icon_sem.csv', index=False)

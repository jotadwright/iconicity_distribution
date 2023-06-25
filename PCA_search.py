import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from matplotlib import pyplot as plt


def reduce_dim(vector, n):
    """Reduces dimensionality of a vector to n via PCA"""
    pca = PCA(n_components=n)
    nd_vec = pca.fit_transform(vector)
    return nd_vec


def lin_reg(X, y):
    # Set semantic vector dimensions as independent variables and iconicity rating
    # as dependent variable for linear regression
    # Split data in 70/30 test/train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

    # Load and fit the linear regression model to the training set.
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)

    # Get predictions
    y_pred = lm.predict(X_test)

    # Get RMSE

    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    return rootMeanSqErr


# Load iconicity dataset
df = pd.read_pickle('data/iconicity_vectors.pkl')

# Create empty list to save each model's RMSE and define number of dimensions to test
dim_rmse = []
dims = range(1, 301)

# Iterate over number of dimesnions, reduce dimensionality with PCA, create LM model and save RMSE
for n in dims:
    sem_vec_red = reduce_dim(df['sem_vec'].to_list(), n)
    print(f"No. Dimensions {n}")
    rmse = lin_reg(sem_vec_red, df['Iconicity'])
    print('\n')
    dim_rmse.append(rmse)

# Plot number of dimensions/RMSE
plt.plot(dims, dim_rmse)
plt.xlabel("No. Dimensions (PCA)")
plt.ylabel("RMSE")
plt.show()

# Save to df and CSV
d = {'Dimension': dims, 'RMSE': dim_rmse}
output = pd.DataFrame(d)
output.to_csv('PCA.csv', index=False)

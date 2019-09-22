import utils

data = x_train, x_test, y_train, y_test = utils.import_wine(n_samples=10000, y_transform=None)

# FINDING k

for k in range(25, 44):
    utils.knn(*data, n_neighbors=k)

# INFLUENCE OF THE WEIGHTS

utils.knn(*data, n_neighbors=31, weights='distance')

# INFLUENCE OF THE METRICS

metrics = ['manhattan', 'chebyshev']

for metric in metrics:
    utils.knn(*data, n_neighbors=31, metric=metric)

# BEST MODEL

data = x_train, x_test, y_train, y_test = utils.import_wine(y_transform=None)

utils.knn(*data, n_neighbors=31)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plot

import utils

data = x_train, x_test, y_train, y_test = utils.import_adult(normalize=True)

# FINDING k

for k in range(18, 24):
    utils.knn(*data, n_neighbors=k)

# INFLUENCE OF THE WEIGHTS

utils.knn(*data, n_neighbors=20, weights='distance')

# INFLUENCE OF THE METRICS

metrics = ['manhattan', 'chebyshev']

for metric in metrics:
    utils.knn(*data, n_neighbors=20, metric=metric)

# BEST MODEL

data = x_train, x_test, y_train, y_test = utils.import_wine(y_transform=None)

utils.knn(*data, n_neighbors=20)

# LEARNING CURVE

clf = KNeighborsClassifier(n_neighbors=20)
x = []
train = []
test = []
for i in [0.02, 0.1, 1, 2, 3, 5, 10, 15, 25, 32]:
    index = int(1000 * i)
    x.append(index)
    clf.fit(x_train[:index], y_train[:index].values.ravel())
    train.append(clf.score(x_train[:index], y_train[:index].values.ravel()))
    test.append(clf.score(x_test, y_test.values.ravel()))

plot.style.use('seaborn-darkgrid')
plot.title('kNN learning curve')
plot.xlabel('Training set size')
plot.ylabel('Score (%)')
plot.plot(x, train)
plot.plot(x, test)
plot.legend(['Training set', 'Testing set'], loc='upper right')
plot.show()

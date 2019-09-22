from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plot

import utils

data = x_train, x_test, y_train, y_test = utils.import_adult()

# WITHOUT PRUNING

utils.dt_pruning(*data)

# PRUNING MANUALLY

utils.dt_pruning(*data, 10, 5)
utils.dt_pruning(*data, 50, 10)
utils.dt_pruning(*data, 100, 20)
utils.dt_pruning(*data, 1000, 50)

# PRUNING WITH 10-FOLD CROSS-VALIDATION

best = utils.dt_crossval(*data, n_leaf_range=range(80, 90), n_depth_range=range(12, 15))

utils.export_graph(best, "adult_tree", features_names=x_train.keys())

# LEARNING CURVE

clf = DecisionTreeClassifier(max_leaf_nodes=80, max_depth=10)
x = []
train = []
test = []
for i in range(1, 320):
    index = int(100 * i)
    x.append(index)
    clf.fit(x_train[:index], y_train[:index])
    train.append(clf.score(x_train[:index], y_train[:index]))
    test.append(clf.score(x_test, y_test))

plot.style.use('seaborn-darkgrid')
plot.title('Decision trees learning curve')
plot.xlabel('Training set size')
plot.ylabel('Score (%)')
plot.plot(x, train)
plot.plot(x, test)
plot.legend(['Training set', 'Testing set'], loc='upper right')
plot.show()

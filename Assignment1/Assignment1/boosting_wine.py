from sklearn.tree import DecisionTreeClassifier

import utils
import numpy as np
import matplotlib.pyplot as plot

data = x_train, x_test, y_train, y_test = utils.import_wine(n_samples=5000, y_transform=None)

# WITHOUT TUNING

base_clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_leaf_nodes=20000, max_depth=50)
utils.boosting(*data, base_clf, verbose=True)

# 10-FOLD CROSSVALIDATION ON THE LEARNING RATE

scores = []
learning_rates = []

for learning_rate in np.linspace(1.3, 1.45, 5):
    print(learning_rate)
    learning_rates.append(learning_rate)
    scores.append(utils.boosting_crossval(*data, base_clf, learning_rate=learning_rate)[1])

plot.style.use('seaborn-darkgrid')
plot.title('Influence of the learning rate on boosting')
plot.xlabel('Learning rate')
plot.ylabel('Score')
plot.plot(learning_rates, scores)
plot.show()

# 10-FOLD CROSSVALIDATION ON THE NUMBER OF ESTIMATORS

scores = []
n_estimators = []

for n_estimator in range(165, 176, 3):
    print(n_estimator)
    n_estimators.append(n_estimator)
    scores.append(utils.boosting_crossval(*data, base_clf, learning_rate=1.37, n_estimators=n_estimator)[1])

plot.style.use('seaborn-darkgrid')
plot.title('Influence of the number of estimators on boosting')
plot.xlabel('Number of estimators')
plot.ylabel('Score')
plot.plot(n_estimators, scores)
plot.show()

# 10-FOLD CROSSVALIDATION ON THE LEARNING RATE TO DOUBLE CHECK

scores = []
learning_rates = []

for learning_rate in np.linspace(1.31, 1.43, 7):
    print(learning_rate)
    learning_rates.append(learning_rate)
    scores.append(utils.boosting_crossval(*data, base_clf, learning_rate=learning_rate, n_estimators=165)[1])

plot.style.use('seaborn-darkgrid')
plot.title('Influence of the learning rate on boosting')
plot.xlabel('Learning rate')
plot.ylabel('Score')
plot.plot(learning_rates, scores)
plot.show()

# DISPLAYING THE CONFUSION MATRIX USING THE BEST HYPERPARAMETERS

data = x_train, x_test, y_train, y_test = utils.import_wine(y_transform=None)
clf, score = utils.boosting(*data, base_clf)

classes = [f'{80 + 4 * k} - {80 + 4 * (k + 1)}' for k in range(5)]
y_pred = clf.predict(x_test)

utils.display_cm(y_test, y_pred, classes)

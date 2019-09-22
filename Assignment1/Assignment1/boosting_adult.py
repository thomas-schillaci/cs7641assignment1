from sklearn.tree import DecisionTreeClassifier

import utils
import numpy as np
import matplotlib.pyplot as plot

data = x_train, x_test, y_train, y_test = utils.import_adult()

# WITHOUT TUNING

base_clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_leaf_nodes=6, max_depth=5)
utils.boosting(*data, base_clf, export_tree=True)

# 10-FOLD CROSSVALIDATION ON THE LEARNING RATE

scores = []
learning_rates = []

for learning_rate in np.linspace(0.1, 0.3, 5):
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

for n_estimator in range(100, 141, 5):
    print(n_estimator)
    n_estimators.append(n_estimator)
    scores.append(utils.boosting_crossval(*data, base_clf, learning_rate=0.2, n_estimators=n_estimator)[1])

plot.style.use('seaborn-darkgrid')
plot.title('Influence of the number of estimators on boosting')
plot.xlabel('Number of estimators')
plot.ylabel('Score')
plot.plot(n_estimators, scores)
plot.show()

# 10-FOLD CROSSVALIDATION ON THE LEARNING RATE TO DOUBLE CHECK

scores = []
learning_rates = []

for learning_rate in np.linspace(0.1, 0.3, 7):
    print(learning_rate)
    learning_rates.append(learning_rate)
    scores.append(utils.boosting_crossval(*data, base_clf, learning_rate=learning_rate, n_estimators=120)[1])

plot.style.use('seaborn-darkgrid')
plot.title('Influence of the learning rate on boosting')
plot.xlabel('Learning rate')
plot.ylabel('Score')
plot.plot(learning_rates, scores)
plot.show()

# DISPLAYING THE CONFUSION MATRIX USING THE BEST HYPERPARAMETERS

clf, score = utils.boosting_crossval(*data, base_clf, learning_rate=0.2, n_estimators=120, verbose=True)
classes = [k for k in range(2)]
utils.display_cm(y_test, clf.predict(x_test), classes)

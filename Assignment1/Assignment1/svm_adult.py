import utils
import numpy as np

data = x_train, x_test, y_train, y_test = utils.import_adult()

# FINDING THE BEST KERNEL AND THE BEST PENALTY C

for C in np.linspace(0.0282, 0.0318, 10):
    utils.svm_crossval(*data, kernel='sigmoid', C=C)

for C in np.linspace(3.25, 3.75, 10):
    utils.svm_crossval(*data, kernel='rbf', C=C)

# BEST MODEL

utils.svm_crossval(*data, kernel='rbf', C=3.5)

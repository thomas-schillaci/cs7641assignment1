import utils
import numpy as np

data = x_train, x_test, y_train, y_test = utils.import_wine(n_samples=10000, y_transform=None)

# INFLUENCE OF THE PENALTY C ON THE SIGMOID AND RBF KERNELS

for C in np.linspace(0.003, 0.012, 10):
    utils.svm_crossval(*data, kernel='sigmoid', C=C)

for C in np.linspace(1.48, 1.75, 10):
    utils.svm_crossval(*data, kernel='rbf', C=C)

# BEST MODEL

utils.svm_crossval(*data, kernel='rbf', C=1.61)

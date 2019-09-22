import utils

data = x_train, x_test, y_train, y_test = utils.import_wine()

# WITHOUT PRUNING

utils.dt_pruning(*data)

# PRUNING MANUALLY

utils.dt_pruning(*data, 1000, 40)
utils.dt_pruning(*data, 10000, 70)
utils.dt_pruning(*data, 40000, 90)
utils.dt_pruning(*data, 45000, 95)

# PRUNING WITH 10-FOLD CROSS-VALIDATION

clf = utils.dt_crossval(*data, n_leaf_range=range(41250, 41751, 100), n_depth_range=range(100, 101))

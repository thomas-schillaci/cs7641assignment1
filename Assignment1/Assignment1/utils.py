import io

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import matplotlib.pyplot as plot
import numpy as np


def import_wine(n_samples=100000, y_transform='get_dummies'):
    dataset1 = pd.read_csv('../wine-reviews/winemag-data-130k-v2.csv')[:n_samples]
    dataset2 = pd.read_csv('../wine-reviews/winemag-data_first150k.csv')[:n_samples]

    x = pd.concat([dataset1, dataset2], ignore_index=True, sort=False)

    del dataset1
    del dataset2

    x = x[x['country'] != '']
    x = x[x['province'] != '']
    x = x[x['variety'] != '']
    x = x[x['description'] != '']

    x = x[['country', 'price', 'province', 'variety', 'description', 'points']]
    x = x.dropna()
    y = x[['points']]
    x = x[['country', 'price', 'province', 'variety', 'description']]

    x = pd.get_dummies(x, columns=['country', 'province', 'variety'])
    x['description'] = x.apply(lambda s: len(s['description']), axis=1)
    x['description'] = (x['description'] - x['description'].min()) / (x['description'].max() - x['description'].min())

    y['points'] = pd.cut(y['points'], 5, labels=[k for k in range(5)])

    if y_transform == 'to_categorical':
        from keras.utils import to_categorical
        y = to_categorical(y)
    elif y_transform == 'get_dummies':
        y = pd.get_dummies(y, columns=['points'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    del x
    del y
    return x_train, x_test, y_train, y_test


def import_adult(use_to_categorical=False, n_samples=32561, normalize=False):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50k']

    train = pd.read_csv('../adult/adult-data.csv', names=names)[:n_samples]
    test = pd.read_csv('../adult/adult-test.csv', names=names)
    test = test.drop([0])

    x_train = train.drop('>50k', 1)
    x_test = test.drop('>50k', 1)

    y_train = train[['>50k']]
    y_test = test[['>50k']]

    for i in range(len(x_train)):
        for key in x_train.keys():
            if x_train.at[i, key] == '?':
                x_train = x_train.drop([i])
                y_train = y_train.drop([i])
                i -= 1
                break

    for i in range(1, len(x_test)):
        for key in x_test.keys():
            if x_test.at[i, key] == '?':
                x_test = x_test.drop([i])
                y_test = y_test.drop([i])
                i -= 1
                break

    x_train = pd.get_dummies(
        x_train,
        columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'native-country']
    )
    x_test = pd.get_dummies(
        x_test,
        columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'native-country']
    ).astype(int)

    tmp = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    y_train['>50k'] = pd.factorize(y_train['>50k'])[0]
    y_test['>50k'] = pd.factorize(y_test['>50k'])[0]

    pd.options.mode.chained_assignment = tmp

    if use_to_categorical:
        from keras.utils import to_categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    for train_key in x_train.keys():
        if train_key not in x_test.keys():
            x_train = x_train.drop(train_key, axis=1)

    for test_key in x_test.keys():
        if test_key not in x_train.keys():
            x_test = x_test.drop(test_key, axis=1)

    if normalize:
        x_train['age'] = (x_train['age'] - x_train['age'].min()) / (x_train['age'].max() - x_train['age'].min())
        x_test['age'] = (x_test['age'] - x_test['age'].min()) / (x_test['age'].max() - x_test['age'].min())
        x_train['fnlwgt'] = (x_train['fnlwgt'] - x_train['fnlwgt'].min()) / (
                x_train['fnlwgt'].max() - x_train['fnlwgt'].min())
        x_test['fnlwgt'] = (x_test['fnlwgt'] - x_test['fnlwgt'].min()) / (
                x_test['fnlwgt'].max() - x_test['fnlwgt'].min())
        x_train['education-num'] = (x_train['education-num'] - x_train['education-num'].min()) / (
                x_train['education-num'].max() - x_train['education-num'].min())
        x_test['education-num'] = (x_test['education-num'] - x_test['education-num'].min()) / (
                x_test['education-num'].max() - x_test['education-num'].min())
        x_train['capital-gain'] = (x_train['capital-gain'] - x_train['capital-gain'].min()) / (
                x_train['capital-gain'].max() - x_train['capital-gain'].min())
        x_test['capital-gain'] = (x_test['capital-gain'] - x_test['capital-gain'].min()) / (
                x_test['capital-gain'].max() - x_test['capital-gain'].min())
        x_train['capital-loss'] = (x_train['capital-loss'] - x_train['capital-loss'].min()) / (
                x_train['capital-loss'].max() - x_train['capital-loss'].min())
        x_test['capital-loss'] = (x_test['capital-loss'] - x_test['capital-loss'].min()) / (
                x_test['capital-loss'].max() - x_test['capital-loss'].min())
        x_train['hours-per-week'] = (x_train['hours-per-week'] - x_train['hours-per-week'].min()) / (
                x_train['hours-per-week'].max() - x_train['hours-per-week'].min())
        x_test['hours-per-week'] = (x_test['hours-per-week'] - x_test['hours-per-week'].min()) / (
                x_test['hours-per-week'].max() - x_test['hours-per-week'].min())

    return x_train, x_test, y_train, y_test


def dt_pruning(x_train, x_test, y_train, y_test, max_leaf_nodes=None, depth=None):
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_leaf_nodes=max_leaf_nodes, max_depth=depth)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test) * 100

    method = 'Without pruning' if max_leaf_nodes is None else 'Pruning manually'

    print(
        f'{method}: score: {round(score, 2)}% for {clf.get_n_leaves()} leaves and a depth of {clf.get_depth()}'
    )


def boosting(
        x_train,
        x_test,
        y_train,
        y_test,
        base_clf,
        learning_rate=1,
        n_estimators=50,
        verbose=True,
        export_tree=False):
    clf = AdaBoostClassifier(
        base_clf,
        random_state=0,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )

    clf.fit(x_train, y_train.values.ravel())
    score = clf.score(x_test, y_test) * 100

    if verbose:
        print(
            f'Boosting score: {round(score, 2)}%'
        )

    if export_tree:
        best_estimator = None
        best_score = 0
        for estimator in clf.estimators_:
            score = estimator.score(x_train, y_train)
            if score > best_score:
                best_score = score
                best_estimator = estimator
        export_graph(best_estimator, "boosting", features_names=x_train.keys())

    return clf, score


def dt_crossval(x_train, x_test, y_train, y_test, n_leaf_range=None, n_depth_range=None):
    maximum = 0
    best = None

    if n_leaf_range is None:
        best = dt_ten_fold(x_train, y_train)[0]

    else:
        for n_leaf in n_leaf_range:
            for n_depth in n_depth_range:
                clf, score = dt_ten_fold(x_train, y_train, max_leaf_nodes=n_leaf, max_depth=n_depth)
                if score > maximum:
                    maximum = score
                    best = clf

    score = best.score(x_test, y_test) * 100

    print(
        f'With cross-validation 10-fold: score={round(score, 2)}% for {best.get_n_leaves()} leaves '
        f'and a depth of {best.get_depth()}'
    )

    return best


def boosting_crossval(
        x_train,
        x_test,
        y_train,
        y_test,
        base_clf,
        verbose=False,
        learning_rate=1,
        n_estimators=50):
    best_clf = None
    best_score = 0

    for k in range(10):
        x_cut, x_val = split_array(x_train, k)
        y_cut, y_val = split_array(y_train, k)

        clf = AdaBoostClassifier(
            base_clf,
            random_state=0,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )

        clf.fit(x_cut, y_cut.values.ravel())
        score = clf.score(x_val, y_val)

        if score > best_score:
            best_score = score
            best_clf = clf

    score = best_clf.score(x_test, y_test)

    if verbose:
        print(
            f'Boosting score with a learning rate of {learning_rate} and {n_estimators} estimators using '
            f'10-fold crossvalidation: {round(score * 100, 2)}%'
        )

    return best_clf, score


def svm_crossval(x_train, x_test, y_train, y_test, kernel='rbf', degree=3, verbose=True, C=1):
    best_clf = None
    best_score = 0

    for k in range(5):
        x_cut, x_val = split_array(x_train, k)
        y_cut, y_val = split_array(y_train, k)

        clf = SVC(
            random_state=0,
            kernel=kernel,
            degree=degree,
            gamma='scale',
            C=C
        )

        clf.fit(x_cut, y_cut.values.ravel())
        score = clf.score(x_val, y_val)

        if score > best_score:
            best_score = score
            best_clf = clf

    score = best_clf.score(x_test, y_test)

    if verbose:
        print(
            f'SVM score with kernel {kernel} and a cost of {C}'
            f' using 5-fold crossvalidation: {round(score * 100, 2)}'
        )

    return best_clf, score


def knn(x_train, x_test, y_train, y_test, n_neighbors=5, weights='uniform', metric='minkowski', verbose=True):
    best_clf = None
    best_score = 0

    for k in range(10):
        x_cut, x_val = split_array(x_train, k)
        y_cut, y_val = split_array(y_train, k)

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )

        clf.fit(x_cut, y_cut.values.ravel())
        score = clf.score(x_val, y_val)

        if score > best_score:
            best_score = score
            best_clf = clf

    score = best_clf.score(x_test, y_test)

    if verbose:
        print(
            f'kNN score with {n_neighbors} neighbors, {weights} weights using the {metric} metric and 10-fold crossvalidation:'
            f' {round(score * 100, 2)}'
        )

    return best_clf, score


def split_array(X, k):
    left = X[:k * len(X) // 10]
    right = X[(k + 1) * len(X) // 10:]
    val = X[k * len(X) // 10:(k + 1) * len(X) // 10]

    return pd.concat([left, right], ignore_index=True, sort=False), val


def dt_ten_fold(x_train, y_train, max_leaf_nodes=None, max_depth=None):
    best_clf = None
    best_score = 0

    for k in range(10):
        x_cut, x_val = split_array(x_train, k)
        y_cut, y_val = split_array(y_train, k)

        clf = DecisionTreeClassifier(
            random_state=0,
            criterion='entropy',
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes
        )

        clf.fit(x_cut, y_cut)
        score = clf.score(x_val, y_val)

        if score > best_score:
            best_score = score
            best_clf = clf

    return best_clf, best_score


def export_graph(clf, name, features_names=None):
    dot_data = io.StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=features_names, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(f'{name}.pdf')


def plot_ann_history(history):
    plot.style.use('seaborn-darkgrid')
    plot.plot(history.history['categorical_accuracy'])
    plot.plot(history.history['val_categorical_accuracy'])
    plot.title('Model accuracy')
    plot.ylabel('Accuracy')
    plot.xlabel('Epoch')
    plot.legend(['Training', 'Validation'], loc='upper left')


def display_cm(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='magma_r')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix of boosting',
           ylabel='True label',
           xlabel='Predicted label')

    plot.setp(ax.get_xticklabels(), rotation=45, ha="right",
              rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    np.set_printoptions(precision=2)

    plot.show()

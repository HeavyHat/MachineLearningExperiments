from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import common
from plotting import LandscapePlotter
from plotting import BoundaryPlotter
from condensed_trees import MimicTreeClassifier

import numpy as np

common.parser.add_argument('--ecount', '-e', dest='ecount', metavar='N', type=int, default=500)


def get_model_definition(n_count, type=None):
    if type is None:
        return AdaBoostClassifier(n_estimators=n_count,
                                  base_estimator=DecisionTreeClassifier(max_depth=1))
    return None

if __name__ == '__main__':
    args = common.parser.parse_args()
    data = common.get_dataset_from_generator(args)
    forestClassifier = get_model_definition(args.ecount)
    training_X, testing_X, training_y, testing_y = train_test_split(data[0], data[1], test_size=0.5)
    forestClassifier.fit(training_X, training_y)
    print np.arange(1, 100,2)
    plotter = BoundaryPlotter(forestClassifier, resolution=800)
    colours = [(1 - target, 0, target) for target in testing_y]
    figure, plot = plotter.plot(training_X, testing_X)
    x = [item[0] for item in testing_X]
    y = [item[1] for item in testing_X]
    predictions = forestClassifier.predict(testing_X)
    plt.scatter(x, y, c=colours)
    plt.text(max(x)+(0.1*max(x)), max(y)+(0.1*max(y)), r'Accuracy = %0.5f' % accuracy_score(testing_y, predictions))
    mimicTree = MimicTreeClassifier(forestClassifier)
    mimicTree.fit(testing_X)
    boundaryplotter = BoundaryPlotter(mimicTree, resolution=800)
    boundaryplotter.plot(training_X, testing_X)
    predictions = mimicTree.predict(testing_X)
    plt.scatter(x, y, c=colours)
    plt.text(max(x) + (0.1 * max(x)), max(y) + (0.1 * max(y)),
             r'Accuracy = %0.5f' % accuracy_score(testing_y, predictions))
    plt.show()



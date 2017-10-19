from sklearn.ensemble import RandomForestClassifier
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import common
from plotting import LandscapePlotter
import numpy as np

common.parser.add_argument('--ecount', '-e', dest='ecount', metavar='N', type=int, default=10)


def get_model_definition(n_count, criterion='gini', type=None):
    if type is None:
        return RandomForestClassifier(n_estimators=n_count, criterion=criterion)
    return None

if __name__ == '__main__':
    args = common.parser.parse_args()
    data = common.get_dataset_from_generator(args)
    forestClassifier = get_model_definition(args.ecount)
    training_X, testing_X, training_y, testing_y = train_test_split(data[0], data[1], test_size=0.5)
    forestClassifier.fit(training_X, training_y)
    print np.arange(1, 100,2)
    plotter = LandscapePlotter(np.arange(1, 100, 2))
    colours = [(1 - target, 0, target) for target in testing_y]
    plot = plotter.plot(get_model_definition, data[0], data[1])
    plt.show()

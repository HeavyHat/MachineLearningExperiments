from sklearn.svm import SVC
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import common
from plotting import BoundaryPlotter

common.parser.add_argument('-C', dest='C', metavar='N', type=float, default=0.2)
common.parser.add_argument('--gamma', '-g', dest='gamma', metavar='N', type=float, default=0.2)


def get_model_definition(C, gamma):
    return SVC(C=C, gamma=gamma)

if __name__ == '__main__':
    args = common.parser.parse_args()
    data = common.get_dataset_from_generator(args)
    forestClassifier = get_model_definition(10, 4)
    training_X, testing_X, training_y, testing_y = train_test_split(data[0], data[1], test_size=0.5)
    forestClassifier.fit(training_X, training_y)
    plotter = BoundaryPlotter(forestClassifier, resolution=1000)
    colours = [(1 - target, 0, target) for target in testing_y]
    figure, plot = plotter.plot(training_X, testing_X)
    plot.scatter(testing_X[:, 0], testing_X[:, 1], c=colours)
    plt.show()

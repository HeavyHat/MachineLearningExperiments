import pandas as pd
from sklearn.datasets import *
from sklearn.svm import SVC
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
from plotting import BoundaryPlotter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', '-c', dest=count', metavar='C', type=int, default=1000)
    parser.add_argument('--noise', '-n', dest='noise', metavar='N', type=float, default=0.2)
    parser.add_argument('--ecount', '-e', dest='ecount', metavar='N', type=int, default=10)
    parser.add_argument('--dataset', '-d', dest='dataset', metavar='N', type=object)
    args = parser.parse_args()
    return args


def get_dataset_from_generator(count, noise, data_generator='make_moons'):
    data = globals()[data_generator](count, noise=noise)
    return data


def get_model_definition(C, gamma):
    return SVC(C=C, gamma=gamma)

if __name__ == '__main__':
    args = parse_arguments()
    if args.dataset is not None:
        data = get_dataset_from_generator(args.count, args.noise, data_generator=args.dataset)
    else:
        data = get_dataset_from_generator(args.count, args.noise)
    forestClassifier = get_model_definition(10, 4)
    training_X, testing_X, training_y, testing_y = train_test_split(data[0], data[1], test_size=0.5)
    forestClassifier.fit(training_X, training_y)
    plotter = BoundaryPlotter(forestClassifier, resolution=1000)
    colours = [(1 - target, 0, target) for target in testing_y]
    figure, plot = plotter.plot(training_X, testing_X)
    plot.scatter(testing_X[:, 0], testing_X[:, 1], c=colours)
    plt.show()

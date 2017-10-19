from sklearn.datasets import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', dest='dataset', metavar='N', default='make_moons')
parser.add_argument('--count', '-c', dest='count', metavar='C', type=int, default=10000)
parser.add_argument('--noise', '-n', dest='noise', metavar='N', type=float, default=0.1)


def get_dataset_from_generator(args):
    data_generator = args.dataset
    if data_generator == 'make_multilabel_classification':
        return globals()[data_generator](args.count)
    return globals()[data_generator](args.count, noise=args.noise)
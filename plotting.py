import matplotlib.pyplot as plt
from sklearn.base import clone
import numpy as np


class BoundaryPlotter:

    def __init__(self, model, resolution=100):
        self.model = model
        self.resolution = resolution

    def plot(self,training_x, testing_x):
        fulldata = training_x + testing_x
        x_min, x_max, y_min, y_max = self.__get_min_max(fulldata)
        x_resolution = (x_max - x_min) / self.resolution
        y_resolution = (y_max - y_min) / self.resolution
        xx, yy = np.meshgrid(np.arange(x_min, x_max, x_resolution),
                             np.arange(y_min, y_max, y_resolution))
        figure, axarr = plt.subplots()
        z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        axarr.contourf(xx, yy, z, alpha=0.4)
        return figure, axarr

    def __get_min_max(self, data):
        y, x = data.shape
        x = [item[0] for item in data]
        y = [item[1] for item in data]
        x_min, x_max = min(x) - 0.1 * max(x), max(x) + 0.1 * max(x)
        y_min, y_max = min(y) - 0.1 * max(y), max(y) + 0.1 * max(y)
        return x_min, x_max, y_min, y_max

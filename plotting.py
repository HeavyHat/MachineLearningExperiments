import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.base import clone
from scipy.interpolate import interp2d
import numpy as np


class BoundaryPlotter:

    def __init__(self, model, resolution=100):
        self.model = model
        self.resolution = resolution

    def plot(self,training_x, testing_x):
        fulldata = training_x + testing_x
        x_min, x_max, y_min, y_max = self.__get_min_max(fulldata)
        x_resolution, y_resolution = self.__get_resolutions(x_min, x_max, y_min, y_max)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, x_resolution),
                    np.arange(y_min, y_max, y_resolution))
        figure, axarr = plt.subplots()
        z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        axarr.contourf(xx, yy, z, alpha=0.3)
        return figure, axarr

    def __get_min_max(self, data):
        y, x = data.shape
        x = [item[0] for item in data]
        y = [item[1] for item in data]
        x_min, x_max = min(x) - 0.1 * max(x), max(x) + 0.1 * max(x)
        y_min, y_max = min(y) - 0.1 * max(y), max(y) + 0.1 * max(y)
        return x_min, x_max, y_min, y_max

    def __get_resolutions(self, x_min, x_max, y_min, y_max):
        x_resolution = (x_max - x_min) / self.resolution
        y_resolution = (y_max - y_min) / self.resolution
        return x_resolution, y_resolution


class Landscape3DPlotter:

    def __init__(self, resolution, x_function, y_function, z_function):
        self.resolution = resolution
        self.x_function = x_function
        self.y_function = y_function
        self.z_function = z_function


    def plot(self, model_function):
        X = self.x_function(range(self.resolution))
        Y = self.y_function(range(self.resolution))
        Z = list()
        for x in X:
            for y in Y:
                model, = model_function(x, y)
                z = self.z_function(model, x, y)
                Z.append(z)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z)
        return ax



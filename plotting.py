import matplotlib.pyplot as plt
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import cross_val_score
else:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
import numpy as np
import cProfile, pstats, StringIO


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
        axarr.contourf(xx, yy, z, alpha=0.3, antialiased=True)

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


def default_cross_val_score(m_func, param, x, y):
    cross_validator = StratifiedKFold(n_splits=5, random_state=1)
    return 1 - cross_val_score(m_func(param), x, y, cv=cross_validator).mean()

class LandscapePlotter:

    def __init__(self, x_iterator, y_generator=default_cross_val_score):
        self.x_iterator = x_iterator
        self.y_generator= y_generator


    def plot(self, model_function, data_X, data_y, axis=None, profile=False):
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        X = self.x_iterator
        Y = [self.y_generator(model_function, item, data_X, data_y) for item in self.x_iterator]
        print X
        print Y
        if axis is None:
            fig, ax = plt.subplots()
        else:
            ax = axis
        ax.plot(X, Y)
        if profile:
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
        return ax


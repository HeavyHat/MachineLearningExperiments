import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data = make_moons(1000, noise=0.2)


forestClassifier = RandomForestClassifier(n_estimators=100, )
training_X, testing_X, training_y, testing_y = train_test_split(data[0], data[1], test_size=0.5)
forestClassifier.fit(training_X, training_y)
colours = [(1-target,0,target) for target in testing_y]
x = [item[0] for item in testing_X]
y = [item[1] for item in testing_X]
x_min, x_max = min(x) - 1, max(x) + 1
y_min, y_max = min(y) - 1, max(y) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
f, axarr = plt.subplots()
z = forestClassifier.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
axarr.contourf(xx, yy, z, alpha=0.4)
axarr.scatter(x, y, c=colours)
plt.show()

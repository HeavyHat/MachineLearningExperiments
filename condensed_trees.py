from sklearn.tree import DecisionTreeClassifier
from scipy.interpolate import griddata

class MimicTreeClassifier:

    def __init__(self, model):
        self.__model = model
        self.__condensed_model = DecisionTreeClassifier(max_depth=None,
                                                        min_samples_split=2,
                                                        min_samples_leaf=1)

    def fit(self, training_X):
        predictions = self.__model.predict(training_X)
        self.__condensed_model.fit(training_X, predictions)

    def predict(self, testing_X):
        return self.__condensed_model.predict(testing_X)
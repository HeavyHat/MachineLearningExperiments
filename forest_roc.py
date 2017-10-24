from sklearn.tree import DecisionTreeClassifier
from plotting import LandscapePlotter
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip


def get_data_set():
    n = 1000
    hastie = make_hastie_10_2(n_samples=n, random_state=1)
    categorical = make_multilabel_classification(n_classes=1, n_samples=n, n_features=3, n_labels=3, random_state=1)
    oh_encoder = OneHotEncoder()
    oh_encoder.fit(categorical[0])
    categorical_one_hot = oh_encoder.transform(categorical[0])
    hastie_reshape = np.reshape(hastie[1], categorical[1].shape)
    answers = np.concatenate((categorical[1], hastie_reshape), axis=1)
    answers = [str(item[0]) + str(item[1]) for item in answers]
    encoder = LabelEncoder()
    encoder.fit(answers)
    answers = encoder.transform(answers)
    data = (np.concatenate((categorical_one_hot.toarray(), hastie[0]), axis=1),
            answers)
    return data

def get_decision_tree(x):
    return DecisionTreeClassifier(random_state=1)

def get_random_forest(x):
    return RandomForestClassifier(n_estimators=x, random_state=1, n_jobs=-1)

def get_adaboost_classifier(x):
    return AdaBoostClassifier(n_estimators=x, random_state=1, algorithm='SAMME')

def get_extratrees_classifier(x):
    return ExtraTreesClassifier(n_estimators=x, random_state=1, n_jobs=-1)

def get_gradientboosted_classifier(x):
    return GradientBoostingClassifier(n_estimators=x, random_state=1)

if __name__ == '__main__':

    models = {
        'Decision Tree' : get_decision_tree,
        'Random Forest' : get_random_forest,
        'Adaboosted Classifier' : get_adaboost_classifier,
        'ExtraTrees Classifier' : get_extratrees_classifier,
        # 'Gradient Boosted Classifier' : get_gradientboosted_classifier
    }
    dataset = get_data_set()
    plotter = LandscapePlotter(np.arange(1,201,10))
    plot = None
    labels = list()
    for model in models:
        print('Running ' + model)
        plot = plotter.plot(models[model], dataset[0], dataset[1], axis=plot)
        labels.append(model)
        print('Finished ' + model)
    plot.legend(labels)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.title('Error Rate against the Number of Estimators in Ensemble Methodology')
    plt.axis([1, 201, 0, 0.5])
    plt.grid(True)
    plt.show()

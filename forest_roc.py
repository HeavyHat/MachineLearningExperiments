from sklearn.tree import DecisionTreeClassifier
from plotting import LandscapePlotter
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn
if sklearn.__version__ < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

def get_data_set(include_categorical=False, one_hot_encoding=False, n=1000):

    hastie = get_continuous_data_set(n)
    if include_categorical:
        categorical = get_categorical_data_set(n)
        if one_hot_encoding:
            oh_encoder = OneHotEncoder()
            oh_encoder.fit(categorical[0])
            cat_data = oh_encoder.transform(categorical[0]).toarray()
        else:
            cat_data = categorical[0]
        hastie_reshape = np.reshape(hastie[1], categorical[1].shape)
        answers = np.concatenate((categorical[1], hastie_reshape), axis=1)
        answers = [str(item[0]) + str(item[1]) for item in answers]
        encoder = LabelEncoder()
        encoder.fit(answers)
        answers = encoder.transform(answers)
        data = (np.concatenate((cat_data, hastie[0]), axis=1),
                answers)
        return data
    return hastie

def get_categorical_data_set(n):
    categorical = make_multilabel_classification(n_classes=1, n_samples=n, n_features=3, n_labels=3, random_state=1,
                                                 allow_unlabeled=False)
    return categorical

def get_continuous_data_set(n):
    hastie = make_hastie_10_2(n_samples=n, random_state=1)
    return hastie

def get_decision_tree(x):
    return DecisionTreeClassifier(random_state=1)

def get_random_forest(x):
    return RandomForestClassifier(n_estimators=x, random_state=1, n_jobs=-1)

def get_adaboost_classifier(x):
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=x, random_state=1, algorithm='SAMME.R')

def get_extratrees_classifier(x):
    return ExtraTreesClassifier(n_estimators=x, random_state=1, n_jobs=-1)

def get_gradientboosted_classifier(x):
    return GradientBoostingClassifier(n_estimators=x, random_state=1)

if __name__ == '__main__':

    start = 1
    end = 500
    step = 100


    models = {
        'Decision Tree' : get_decision_tree,
        'Random Forest' : get_random_forest,
        'Adaboosted Classifier' : get_adaboost_classifier,
        'ExtraTrees Classifier' : get_extratrees_classifier,
        'Gradient Boosted Classifier' : get_gradientboosted_classifier
    }
    dataset = get_data_set(include_categorical=True)
    x = np.append([start], np.arange(step+start if step == start else step,end+1,step))
    print x
    plotter = LandscapePlotter(x)
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
    plt.axis([start, end, 0, 0.5])
    plt.grid(True)

    for model in models:
        training_X, testing_X, training_y, testing_y = train_test_split(dataset[0], dataset[1], test_size=0.5,
                                                                        stratify=dataset[1])
        classifier = models[model](end)
        classifier.fit(training_X, training_y)
        predictions = classifier.predict(training_X)
        print predictions
        training_error = accuracy_score(training_y, predictions)
        classifier.fit(training_X, training_y)
        predictions_valid = classifier.predict(testing_y)
        print predictions_valid
        validation_error = accuracy_score(testing_y, predictions_valid)
        print 'Model: %s' % model
        print 'Training Error: %0.5f' % training_error
        print 'Validation Error: %0.5f' % validation_error
        print 'Fit Ratio: %0.5f' % (validation_error/training_error)
        print '----'

    tpr_models = dict()
    fpr_models = dict()
    plt.figure()

    for model in models:
        training_X, testing_X, training_y, testing_y = train_test_split(dataset[0], dataset[1], test_size=0.5,
                                                                        stratify=dataset[1])
        classifier = models[model](end)
        classifier.fit(training_X, training_y)
        y_predict = classifier.predict_proba(testing_X)[:, 1]
        fpr, tpr, _ = roc_curve(testing_y, y_predict)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s ROC curve (area = %0.5f)' % (model, area))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()



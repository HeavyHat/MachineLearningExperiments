from sklearn.tree import DecisionTreeClassifier
from plotting import LandscapePlotter
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
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

max_depth = None
max_features = 'auto'
min_samples_leaf = 1
min_samples_split = 2
random_state = 1
max_e = 100

def get_data_set(include_categorical=False, one_hot_encoding=False, n=5000):
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

def get_decision_tree_with_n_estimators(x):
    return DecisionTreeClassifier(max_depth=max_depth,
                                  max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  random_state=random_state)

def get_random_forest_with_n_estimators(x):
    return RandomForestClassifier(max_depth=max_depth,
                                  n_estimators=x,
                                  random_state=random_state,
                                  max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_jobs=-1)

def get_adaboost_classifier_with_n_estimators(x):
    return AdaBoostClassifier(base_estimator=get_decision_tree_with_n_estimators(x),
                              n_estimators=x,
                              random_state=random_state,
                              learning_rate=0.01,
                              algorithm='SAMME.R')

def get_extratrees_classifier_with_n_estimators(x):
    return ExtraTreesClassifier(max_depth=max_depth,
                                n_estimators=x,
                                random_state=random_state,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1)

def get_gradientboosted_classifier_with_n_estimators(x):
    return GradientBoostingClassifier(max_depth=1,
                                      n_estimators=x,
                                      max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      learning_rate=0.01,
                                      random_state=random_state)

def get_bagging_classifier_with_n_estimators(x):
    return BaggingClassifier(base_estimator=get_decision_tree_with_n_estimators(x),
                             n_estimators=x,
                             random_state=random_state)

def get_decision_tree_with_n_depth(x):
    return DecisionTreeClassifier(max_depth=x,
                                  max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  random_state=random_state)

def get_random_forest_with_n_depth(x):
    return RandomForestClassifier(max_depth=x,
                                  n_estimators=max_e,
                                  random_state=random_state,
                                  max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_jobs=-1)

def get_adaboost_classifier_with_n_depth(x):
    return AdaBoostClassifier(base_estimator=get_decision_tree_with_n_depth(x),
                              n_estimators=max_e,
                              random_state=random_state,
                              learning_rate=0.5,
                              algorithm='SAMME.R')

def get_extratrees_classifier_with_n_depth(x):
    return ExtraTreesClassifier(max_depth=x,
                                n_estimators=max_e,
                                random_state=random_state,
                                max_features=max_features,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1)

def get_gradientboosted_classifier_with_n_depth(x):
    return GradientBoostingClassifier(max_depth=x,
                                      n_estimators=max_e,
                                      max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      learning_rate=0.5,
                                      random_state=random_state)

def get_bagging_classifier_with_n_depth(x):
    return BaggingClassifier(base_estimator=get_decision_tree_with_n_depth(x),
                             n_estimators=max_e,
                             random_state=random_state)


def get_decision_tree_with_n_features(x):
    return DecisionTreeClassifier(max_depth=max_depth,
                                  max_features=x,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  random_state=random_state)

def get_random_forest_with_n_features(x):
    return RandomForestClassifier(max_depth=max_depth,
                                  n_estimators=max_e,
                                  random_state=random_state,
                                  max_features=x,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_jobs=-1)

def get_adaboost_classifier_with_n_features(x):
    return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                  max_features=x,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  random_state=random_state),
                              n_estimators=max_e,
                              random_state=random_state,
                              learning_rate=0.5,
                              algorithm='SAMME.R')

def get_extratrees_classifier_with_n_features(x):
    return ExtraTreesClassifier(max_depth=max_depth,
                                n_estimators=max_e,
                                random_state=random_state,
                                max_features=x,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                n_jobs=-1)

def get_gradientboosted_classifier_with_n_features(x):
    return GradientBoostingClassifier(max_depth=1,
                                      n_estimators=max_e,
                                      max_features=x,
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      learning_rate=0.5,
                                      random_state=random_state)

def get_bagging_classifier_with_n_features(x):
    return BaggingClassifier(base_estimator=get_decision_tree_with_n_features(x),
                             n_estimators=max_e,
                             random_state=random_state)

if __name__ == '__main__':

    start = 1
    end = 200
    step = 10

    bagging = {
        'Random Forest',
        'ExtraTrees',
        'Random Subspace',
    }

    boosting = {
        'Adaboosted',
        'Gradient Boosted',
    }

    models = {
        'Decision Tree' : get_decision_tree_with_n_estimators,
        'Random Forest' : get_random_forest_with_n_estimators,
        'Adaboosted' : get_adaboost_classifier_with_n_estimators,
        'ExtraTrees' : get_extratrees_classifier_with_n_estimators,
        'Gradient Boosted' : get_gradientboosted_classifier_with_n_estimators,
        'Random Subspace' : get_bagging_classifier_with_n_estimators
    }
    dataset = get_data_set(n=1000)
    x = np.append([start], np.arange(step+start if step <= start else step,end+1,step))
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
        training_error = accuracy_score(training_y, predictions)
        classifier.fit(training_X, training_y)
        predictions_valid = classifier.predict(testing_X)
        validation_error = accuracy_score(testing_y, predictions_valid)
        print 'Model: %s' % model
        print 'Training Error: %0.5f' % training_error
        print 'Validation Error: %0.5f' % validation_error
        print 'Fit Ratio: %0.5f%%' % (validation_error/training_error)
        print '----'

    tpr_models = dict()
    fpr_models = dict()


    baseline_models = [
        'Decision Tree',
        'Random Forest'
    ]

    for model in models:
        if model in baseline_models:
            continue
        plt.figure()
        training_X, testing_X, training_y, testing_y = train_test_split(dataset[0], dataset[1], test_size=0.5,
                                                                        stratify=dataset[1], random_state=1)
        for baseline in baseline_models + [model]:
            classifier = models[baseline](end)
            classifier.fit(training_X, training_y)
            y_predict = classifier.predict_proba(testing_X)[:, 1]
            fpr, tpr, _ = roc_curve(testing_y, y_predict)
            area = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='%s ROC curve (area = %0.5f)' % (baseline, area))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.legend(loc="lower right")
        plt.grid(True)
    plt.figure()
    start = 1
    end = 20
    step = 1

    models = {
        'Decision Tree': get_decision_tree_with_n_depth,
        'Random Forest': get_random_forest_with_n_depth,
        'Adaboosted': get_adaboost_classifier_with_n_depth,
        'ExtraTrees': get_extratrees_classifier_with_n_depth,
        'Gradient Boosted': get_gradientboosted_classifier_with_n_depth,
        'Random Subspace': get_bagging_classifier_with_n_depth
    }

    x = np.append([start], np.arange(step + start if step <= start else step, end + 1, step))
    print x
    plotter = LandscapePlotter(x)
    plot = None
    labels = list()
    for model in bagging:
        print('Running ' + model)
        plot = plotter.plot(models[model], dataset[0], dataset[1], axis=plot)
        labels.append(model)
        print('Finished ' + model)
    plot.legend(labels)
    plt.xlabel('Maximum base model depth.')
    plt.ylabel('Error Rate')
    plt.title('Error Rate against the Depth of Base Model')
    plt.axis([start, end, 0, 0.5])
    plt.grid(True)
    plt.figure()
    labels = list()
    plot=None
    for model in boosting:
        print('Running ' + model)
        plot = plotter.plot(models[model], dataset[0], dataset[1], axis=plot)
        labels.append(model)
        print('Finished ' + model)
    plot.legend(labels)
    plt.xlabel('Maximum base model depth.')
    plt.ylabel('Error Rate')
    plt.title('Error Rate against the Depth of Base Model')
    plt.axis([start, end, 0, 0.5])
    plt.grid(True)

    for model in models:
        training_X, testing_X, training_y, testing_y = train_test_split(dataset[0], dataset[1], test_size=0.5,
                                                                        stratify=dataset[1])
        classifier = models[model](end)
        classifier.fit(training_X, training_y)
        predictions = classifier.predict(training_X)
        training_error = accuracy_score(training_y, predictions)
        classifier.fit(training_X, training_y)
        predictions_valid = classifier.predict(testing_X)
        validation_error = accuracy_score(testing_y, predictions_valid)
        print 'Model: %s' % model
        print 'Training Error: %0.5f' % training_error
        print 'Validation Error: %0.5f' % validation_error
        print 'Fit Ratio: %0.5f%%' % (validation_error / training_error)
        print '----'

    tpr_models = dict()
    fpr_models = dict()

    baseline_models = [
        'Decision Tree',
        'Random Forest'
    ]

    for model in models:
        if model in baseline_models:
            continue
        plt.figure()
        training_X, testing_X, training_y, testing_y = train_test_split(dataset[0], dataset[1], test_size=0.5,
                                                                        stratify=dataset[1], random_state=1)
        for baseline in baseline_models + [model]:
            classifier = models[baseline](end)
            classifier.fit(training_X, training_y)
            y_predict = classifier.predict_proba(testing_X)[:, 1]
            fpr, tpr, _ = roc_curve(testing_y, y_predict)
            area = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='%s ROC curve (area = %0.5f)' % (baseline, area))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.legend(loc="lower right")
        plt.grid(True)

    plt.figure()

    start = 1
    end = 10
    step = 1

    models = {
        'Decision Tree': get_decision_tree_with_n_features,
        'Random Forest': get_random_forest_with_n_features,
        'Adaboosted': get_adaboost_classifier_with_n_features,
        'ExtraTrees': get_extratrees_classifier_with_n_features,
        'Gradient Boosted': get_gradientboosted_classifier_with_n_features,
        'Random Subspace': get_bagging_classifier_with_n_features
    }

    x = np.append([start], np.arange(step + start if step <= start else step, end + 1, step))

    plotter = LandscapePlotter(x)
    plot = None
    labels = list()
    for model in models:
        print('Running ' + model)
        plot = plotter.plot(models[model], dataset[0], dataset[1], axis=plot)
        labels.append(model)
        print('Finished ' + model)
    plot.legend(labels)
    plt.xlabel('Number of Features')
    plt.ylabel('Error Rate')
    plt.title('Error Rate against the Number of Features in Base Learner')
    plt.axis([start, end, 0, 0.5])
    plt.grid(True)

    next = [
        'Adaboosted',
        'Gradient Boosted',
        'ExtraTrees',
        'Random Forest'
    ]

    for model in next:
        classifier = models[model](10)
        classifier.fit(dataset[0], dataset[1])
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances for %s" % model)
        plt.bar(range(dataset[0].shape[1]), importances[indices],
                color="r", align="center")
        plt.xticks(range(dataset[0].shape[1]), indices)
        plt.xlim([-1, dataset[0].shape[1]])
        plt.xlabel('Feature Number')
        plt.ylabel('Importance')

    plt.show()

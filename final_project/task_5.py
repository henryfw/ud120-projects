import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import grid_search

import task_1, task_2, task_3, task_4


data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
data_dict = task_2.remove_outlier(data_dict)
features_list = task_3.get_feature_list()
my_dataset = task_3.normalize_data(data_dict)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

def run():
    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = task_2.remove_outlier(data_dict)
    get_best_classifier(task_4.get_classifier_dict(), task_3.normalize_data(data_dict))


def get_best_classifier(classifier_dict, my_dataset):

    features_list = task_3.get_feature_list()

    best_clf = None
    for name, clf in classifier_dict.iteritems():
        results = test_classifier(clf, my_dataset, features_list)

        if name == 'dt' :
            #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            parameters = {'min_samples_split': [1, 2, 3, 4, 5, 8, 10, 15, 20]}
            grid = grid_search.GridSearchCV(clf, parameters)
            grid.fit(features, labels)
            print grid.grid_scores_

    return



if __name__ == '__main__':
    run()
import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.tree import DecisionTreeClassifier
import task_1, task_2, task_3, task_4


data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
data_dict = task_2.remove_outlier(data_dict)
features_list = task_3.get_feature_list()
data_dict = task_3.prepare_data(data_dict)
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

def run():
    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = task_2.remove_outlier(data_dict)
    get_best_classifier(task_4.get_classifier_dict(), task_3.prepare_data(data_dict))


def get_best_classifier(classifier_dict, data_dict):

    features_list = task_3.get_feature_list()

    all_results = []
    for name, clf in classifier_dict.iteritems():

        if name == 'nb':
            classifier = clf()
            r = test_classifier(classifier, data_dict, features_list)
            if r is not None:
                r['classifier'] = classifier
                all_results.append(r)

        if name == 'ada':
            for param in  [2, 10, 20, 30] :
                classifier = clf(base_estimator = DecisionTreeClassifier(min_samples_split = param))
                r = test_classifier(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                all_results.append(r)


        if name == 'dt' :
            #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            for param in  [1, 2, 5, 10, 15, 20, 25, 30] :
                classifier = clf(min_samples_split = param)
                r = test_classifier(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                    all_results.append(r)


        if name == 'rf' :
            #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            for param in  [10, 25, 50, 100] :
                classifier = clf(min_samples_split = 20, n_estimators = param)
                r = test_classifier(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                    all_results.append(r)

        if name == 'svm' :
            for param_2 in ('linear', 'rbf') :
                for param in  [10., 100., 1000., 10000.] :
                    classifier = clf(kernel = param_2, C = param)
                    r = test_classifier(classifier, data_dict, features_list)
                    if r is not None:
                        r['classifier'] = classifier
                        all_results.append(r)

    all_results = sorted(all_results, key = lambda x :
        -1 if x is None or x['recall'] < .3 or x['precision'] < .3
        else x['recall'] + x['precision'], reverse = True )

    print "Top 10 Classifiers: "
    pprint.pprint(all_results[0:10])


    with open('task_5_results.pkl', 'w') as f:
        cPickle.dump(all_results, f)

    # hard code the winner in case modified tester file is not used
    best_clf = DecisionTreeClassifier(min_samples_split = 20)

    # save to plk file
    with open('my_classifier.pkl', 'w') as f:
        cPickle.dump(best_clf, f)

    if len(all_results) > 0 :
        # this result depends on using the modified tester file
        return  all_results[0]['classifier']
    else :
        # if the modified tester file is not avail, then hard code the result
        return best_clf



if __name__ == '__main__':
    run()
import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit 

from sklearn.tree import DecisionTreeClassifier
import task_1, task_2, task_3, task_4, task_5_tester


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

    all_results = []
    for name, clf in classifier_dict.iteritems():

        print "Testing " + name

        if name == 'nb':
            classifier = clf()
            r = task_5_tester.test_classifier_custom(classifier, data_dict, features_list)
            if r is not None:
                r['classifier'] = classifier
                all_results.append(r)

        if name == 'ada':
            for param in  [2, 10, 20, 30] :
                classifier = clf(base_estimator = DecisionTreeClassifier(min_samples_split = param))
                r = task_5_tester.test_classifier_custom(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                all_results.append(r)


        if name == 'dt' :
            #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            for param in  [1, 2, 5, 10, 15, 20, 25, 30] :
                classifier = clf(min_samples_split = param)
                r = task_5_tester.test_classifier_custom(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                    all_results.append(r)


        if name == 'rf' :
            #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            for param in  [10, 25, 50, 100] :
                classifier = clf(min_samples_split = 20, n_estimators = param)
                r = task_5_tester.test_classifier_custom(classifier, data_dict, features_list)
                if r is not None:
                    r['classifier'] = classifier
                    all_results.append(r)

        if name == 'svm' :
            for param_2 in ('linear', 'rbf') :
                for param in  [10., 100., 1000., 10000.] :
                    classifier = clf(kernel = param_2, C = param)
                    r = task_5_tester.test_classifier_custom(classifier, data_dict, features_list)
                    if r is not None:
                        r['classifier'] = classifier
                        all_results.append(r)

    all_results = sorted(all_results, key = lambda x :
        -1 if x is None or x['recall'] < .3 or x['precision'] < .3
        else x['recall'] + x['precision'], reverse = True )

    print "Top 10 Classifiers: "
    pprint.pprint(all_results[0:10])

    csv_data = [ "%s,%.3f,%.3f,%.3f\n" %
        ( str_clf(r['classifier']), r['precision'], r['recall'], r['precision'] + r['recall']) for r in all_results[0:10] ]
    with open("task_5_top_10_data.csv", "w") as f:
        f.writelines(csv_data)

    with open('task_5_results.pkl', 'w') as f:
        cPickle.dump(all_results, f)

    best_clf = all_results[0]['classifier']

    # save to plk file
    with open('my_classifier.pkl', 'w') as f:
        cPickle.dump(best_clf, f)

    return best_clf



def str_clf(clf) :
    return  '"' + str(clf).replace("\n", " ").replace("\r", " ") + '"'

if __name__ == '__main__':
    run()
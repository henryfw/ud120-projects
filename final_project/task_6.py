# in this audacious bonus task, we iterate over all combination of x (17 choose x) features and use decision tree with min_samples_split = 10


import cPickle
import sys
import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from time import time
from sklearn.tree import DecisionTreeClassifier
import task_1, task_2, task_3, task_4


def run() :
    start_time = time()

    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = task_2.remove_outlier(data_dict)

    all_feature_list = task_1.get_all_possible_feature_list()


    all_results = []
    clf = DecisionTreeClassifier(min_samples_split=10)

    total_features = len(all_feature_list)
    total_fields = 5

    # need at least 2 not None fields. for total_field 5, it's 20 choose 5 = 15504. ~6 hours on MBP i7
    for i in range(total_fields - 2):
        all_feature_list.append(None)

    i = [ 0 for i in range(total_fields + 1) ]

    for i[1] in range(1, total_features - (total_fields - 1)) :
        for i[2] in range(i[1] + 1, total_features - (total_fields - 2)) :
            for i[3] in range(i[2] + 1, total_features - (total_fields - 3)) :
                for i[4] in range(i[3] + 1, total_features - (total_fields - 4)):
                    for i[5] in range(i[4] + 1, total_features) :
                        features_list = ['poi']
                        for j in range(1, total_fields + 1):
                            if i[j] is not None:
                                features_list.append(all_feature_list[i[j]])

                        data_dict = task_3.prepare_data(data_dict, features_list, False) # normalized data
                        features_list = task_3.get_feature_list(features_list, False) # get normalized fields
                        data = featureFormat(data_dict, features_list, sort_keys = True)
                        labels, features = targetFeatureSplit(data)

                        r = test_classifier(clf, data_dict, features_list)
                        if r is not None:
                            r['classifier'] = clf
                            r['importances'] = clf.feature_importances_
                            r['feature_list'] = features_list[1:]
                            r['feature_total'] = len(features_list) - 1
                            all_results.append(r)


    all_results = sorted(all_results, key = lambda x :
        -1 if x is None or x['recall'] < .3 or x['precision'] < .3
        else x['recall'] + x['precision'], reverse = True )

    print "Training Time: ", round(time() - start_time, 3), "s"
    print "Samples Tested: " , len(all_results)
    print "Top 10 Feature List: "
    pprint.pprint(all_results[0:10])

    with open('task_6_results_for_%d_vars.pkl' % total_fields, 'w') as f:
        cPickle.dump(all_results, f)


    # do most popular fields
    fields = {}
    for r in all_results[0:10] :
        feature_list = r['feature_list']
        for i in feature_list :
            if i == 'poi':
                continue
            if i not in fields :
                fields[i] = 0
            fields[i] += 1

    print "Most used fields in top 10"
    pprint.pprint(all_results[0:10])

    with open('task_6_results_for_%d_fields.pkl' % total_fields, 'w') as f:
        cPickle.dump(fields, f)


def is_unique_list(x) :
    return len(x) == len(set(x))





if __name__ == '__main__':
    run()
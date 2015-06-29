# we iterate over all combination of 4 features and use decision tree with min_samples_split = 10


import cPickle
import sys
import pprint
sys.path.append("../tools/")
from task_5_tester import test_classifier_custom

from time import time
from sklearn.tree import DecisionTreeClassifier
import task_1, task_2, task_3, task_4


def run() :
    start_time = time()

    all_feature_list = task_1.get_all_possible_feature_list()
    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = task_2.remove_outlier(data_dict)
    data_dict = task_3.prepare_data(data_dict, all_feature_list, False) # normalized data

    for i in range(2):
        all_feature_list.append('_none_' + str(i))

    all_results = []
    completed_hashes = {}
    clf = DecisionTreeClassifier(min_samples_split=10)

    for i1 in all_feature_list[1:] :
        for i2 in all_feature_list[2:] :
            for i3 in all_feature_list[3:] :
                for i4 in all_feature_list[4:] :
                    features_list = ['poi']
                    if i1[0:6] != '_none_': features_list.append(i1)
                    if i2[0:6] != '_none_': features_list.append(i2)
                    if i3[0:6] != '_none_': features_list.append(i3)
                    if i4[0:6] != '_none_': features_list.append(i4)

                    # need to do nCr more efficiently
                    if not is_unique_list(features_list):
                        continue
                    hash = '-'.join(sorted(features_list))
                    if hash in completed_hashes :
                        continue
                    completed_hashes[hash] = 1

                    tmp = len(completed_hashes)
                    if tmp % 500 == 1 :
                        print "Testing %d ..." % tmp

                    features_list = task_3.get_feature_list(features_list, False) # get normalized fields

                    r = None
                    try:
                        r = test_classifier_custom(clf, data_dict, features_list)
                    except ValueError:
                        print features_list, ValueError
                    if r is not None:
                        r['classifier'] = clf
                        r['importances'] = clf.feature_importances_
                        r['feature_list'] = features_list
                        all_results.append(r)


    all_results = sorted(all_results, key = lambda x :
        -1 if x is None or x['recall'] < .3 or x['precision'] < .3
        else x['recall'] + x['precision'], reverse = True )

    print "Training Time: ", round(time() - start_time, 3), "s"
    print "Samples Tested: " , len(all_results)
    print "Top 10 Feature List: "
    pprint.pprint(all_results[0:10])

    with open('task_6_results.pkl', 'w') as f:
        cPickle.dump(all_results, f)

    # do most popular fields
    fields = {}
    field_importances = {}
    for r in all_results[0:10] :
        feature_list = r['feature_list']
        for i in feature_list :
            if i == 'poi':
                continue
            if i not in fields :
                fields[i] = 0
            fields[i] += 1
        for i in range(len(r['importances'])) :
            field = feature_list[i + 1]
            if field not in field_importances :
                field_importances[field] = 0
            field_importances[field] += r['importances'][i]

    fields = sorted(fields.items(), key = lambda x : x[1], reverse = True)
    field_importances = sorted(field_importances.items(), key = lambda x : x[1], reverse = True)

    print "Most used fields/importances in top 10:"
    pprint.pprint(fields)
    pprint.pprint(field_importances)

    return all_results[0]

def is_unique_list(x) :
    return len(x) == len(set(x))





if __name__ == '__main__':
    run()
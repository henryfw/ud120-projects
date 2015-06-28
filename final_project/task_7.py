# in this additional task, we re-run DT mutliple times with varied min_samples_split to get averaged results

import cPickle
import sys
import pprint
sys.path.append("../tools/")
from tester import test_classifier_custom, dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier


features_list = data_dict = cPickle.load(open("my_feature_list.pkl", "r") )
data_dict = cPickle.load(open("my_dataset.pkl", "r") )

all_results = {}

def run():
    for split_num in range(5, 51, 5) :
        clf = DecisionTreeClassifier(min_samples_split=split_num)
        all_results[split_num] = None
        print "Running split_num: %d" % split_num

        total_precision = 0.
        total_accuracy = 0.
        total_recall = 0.
        total_trials = 50

        for _ in range(total_trials) :
            r = test_classifier_custom(clf, data_dict, features_list)
            total_precision += r['precision']
            total_accuracy += r['accuracy']
            total_recall += r['recall']

        all_results[split_num] = {
            'precision' : total_precision / total_trials,
            'accuracy' : total_accuracy / total_trials,
            'recall' : total_recall / total_trials,
        }

    pprint.pprint(all_results)

    with open('task_7_results.pkl', 'w') as f:
        cPickle.dump(all_results, f)

if __name__ == '__main__':
    run()
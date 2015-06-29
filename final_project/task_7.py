# in this additional task, we re-run DT mutliple times with varied min_samples_split to get averaged results

import cPickle
import sys
import pprint
sys.path.append("../tools/")
from task_5_tester import test_classifier_custom
from sklearn.tree import DecisionTreeClassifier


features_list = data_dict = cPickle.load(open("my_feature_list.pkl", "r") )
data_dict = cPickle.load(open("my_dataset.pkl", "r") )

all_results = []

def run():
    best_clf = None
    best_score = 0
    for split_num in range(5, 51, 5) :
        clf = DecisionTreeClassifier(min_samples_split=split_num)

        print "Running DT split_num: %d" % split_num

        total_precision = 0.
        total_accuracy = 0.
        total_recall = 0.
        total_trials = 50

        for _ in range(total_trials) :
            r = test_classifier_custom(clf, data_dict, features_list)
            total_precision += r['precision']
            total_accuracy += r['accuracy']
            total_recall += r['recall']

        item = {
            'split_num' : split_num,
            'precision' : total_precision / total_trials,
            'accuracy' : total_accuracy / total_trials,
            'recall' : total_recall / total_trials,
        }
        all_results.append( item )

        score = item['precision'] + item['recall']
        if best_clf is None or score > best_score :
            best_clf = clf
            best_score = score

    pprint.pprint(all_results)

    csv_data = [ "%d,%f,%f\n" % (i['split_num'], i['precision'], i['recall']) for i in all_results ]
    with open("task_7_data.csv", "w") as f:
        f.writelines(csv_data)

    with open('task_7_results.pkl', 'w') as f:
        cPickle.dump(all_results, f)


    with open('my_classifier.pkl', 'w') as f:
        cPickle.dump(best_clf, f)

    return best_clf


if __name__ == '__main__':
    run()
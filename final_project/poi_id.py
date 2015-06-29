#!/usr/bin/python

import sys
import cPickle # changed to cPickle from pickle for mac
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


import task_1, task_2, task_3, task_4, task_5, task_7

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = task_1.get_feature_list()
# ['poi', 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options', 'fraction_to_shared_with_poi' ]


### Load the dictionary containing the dataset
data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict = task_2.remove_outlier(data_dict)
# removes data_dict['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = task_3.prepare_data(data_dict)
# added 3 new columns ['fraction_from_poi', 'fraction_to_poi', 'fraction_to_shared_with_poi']
# added 23 more new columns with normalized data between 0.0 and 1.0, ie. fraction_from_poi_n, fraction_to_poi_n ...

### Extract features and labels from dataset for local testing
features_list = task_3.get_feature_list()
# new feature_list after normalizing
# ['poi', 'total_payments_n', 'loan_advances_n', 'total_stock_value_n', 'exercised_stock_options_n', 'fraction_to_shared_with_poi_n' ]
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
classifer_dict = task_4.get_classifier_dict()
# returns an array dictionary of classifiers to test
'''
{
    'nb' : GaussianNB,
    'dt' : DecisionTreeClassifier,
    'svm' : SVC,
    'ada' : AdaBoostClassifier,
    'rf' : RandomForestClassifier
}
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
clf = task_5.get_best_classifier(classifer_dict, my_dataset)
# best classifier is Decision Tree with min_samples_split of 25
clf = task_7.run()
# retest DT for different min_samples_split with 50 runs per setting
# best classifier is now Decision Tree with min_samples_split of 30

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)
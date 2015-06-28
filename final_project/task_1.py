import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn import tree
from scipy import stats
from sklearn.decomposition import *

import task_1_get_feature_list, task_2, task_3, task_3_helper


# remove outlier that we find later and add new feature to test pca
all_feature_list =  ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'fraction_from_poi', 'fraction_to_poi', 'fraction_to_shared_with_poi'
        ]
data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
data_dict = task_2.remove_outlier(data_dict)
data_dict = task_3_helper.add_new_feature(data_dict)
data_dict = task_3_helper.normalize(data_dict, all_feature_list)



def get_all_possible_feature_list() :
    return all_feature_list


# fields after investigating with the run() function
def get_feature_list() :
    return task_1_get_feature_list.get_feature_list()


def run() :
    test_pca_eigenvalue()
    test_dt_importance()


def test_pca_eigenvalue() :
    #pprint.pprint( data_dict )
    data = featureFormat(data_dict, all_feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    pca = PCA().fit(features)
    eigenfaces = pca.explained_variance_ratio_

    print all_feature_list[1:]
    print [ "%.5f" % i for i in eigenfaces ]
    print [ "%.5f" % abs(i) for i in pca.components_[0] ]
'''
['0.00825', '0.00689', '0.68172', '0.51338', '0.05344', '0.00500', '0.00689', '0.40506', '0.00057', '0.29293', '0.07376', '0.02762', '0.11211', '0.00017', '0.00000', '0.00000']
# first component is has high abs values for 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options'
'''

def test_dt_importance():
    clf = tree.DecisionTreeClassifier(min_samples_split=10)
    importances = {}
    for _ in range(100):
        data = featureFormat(data_dict, get_all_possible_feature_list(), sort_keys = True)
        labels, features = targetFeatureSplit(data)
        clf.fit(features, labels)

        for i in range(len(clf.feature_importances_)) :
            key = str(get_all_possible_feature_list()[i+1]) + ' ' + str(i+1)
            if key not in importances :
                importances[key] = 0

            importances[key] += clf.feature_importances_[i]


    # sort by importance
    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True )

    pprint.pprint(importances)

    '''
 # after removing 'TOTAL' outlier
[('exercised_stock_options 10', 25.609897292250217),
 ('expenses 9', 19.55805317295998),
 ('fraction_to_poi 16', 15.443537414966045),
 ('deferred_income 7', 10.562238930659962),
 ('fraction_to_shared_with_poi 17', 8.3687479568486491),
 ('total_stock_value 8', 5.2544573384909521),
 ('restricted_stock 13', 4.2721533057667491),
 ('total_payments 3', 3.5785336356764925),
 ('bonus 5', 3.1661375661375666),
 ('long_term_incentive 12', 2.2391534391534389),
 ('salary 1', 1.2698412698412698),
 ('other 11', 0.67724867724867721),
 ('fraction_from_poi 15', 0.0),
 ('loan_advances 4', 0.0),
 ('director_fees 14', 0.0),
 ('restricted_stock_deferred 6', 0.0),
 ('deferral_payments 2', 0.0)]
     '''

    # find correlated fields between important features for task 3
    top_features_index = [10,9,16,7,17]
    text = ""
    for i in top_features_index:
        for j in top_features_index:
            text += "%.3f " % stats.pearsonr(features[i], features[j])[0]
        text += "\n"
    print text
'''
1.000 0.470 0.238 0.524 0.587
0.470 1.000 -0.107 0.070 0.709
0.238 -0.107 1.000 0.386 0.285
0.524 0.070 0.386 1.000 0.620
0.587 0.709 0.285 0.620 1.000
'''



if __name__ == '__main__':
    run()
import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn import tree
from scipy import stats
from sklearn.decomposition import *

# test all feature importance in decision tree
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

my_dataset = cPickle.load(open("final_project_dataset.pkl", "r") )
del my_dataset['TOTAL']# remove TOTAL




# fields after investigating with the run() function
def get_feature_list() :

    # from dt importance: 'exercised_stock_options', 'total_payments', 'bonus'
    # items = ['poi', 'exercised_stock_options', 'total_payments', 'bonus' ]

    # from pca 1st component:  'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options'
    items = ['poi', 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options' ]
    return items


def run() :
    test_pca_eigenvalue()
    test_dt_importance()


def test_pca_eigenvalue() :
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    pca = PCA().fit(features)
    eigenfaces = pca.explained_variance_ratio_

    print [ "%.5f" % i for i in eigenfaces ]
    '''
['0.80547', '0.14734', '0.01642', '0.01472', '0.00748', '0.00439', '0.00204', '0.00125', '0.00074', '0.00010', '0.00005', '0.00001', '0.00000', '0.00000']
# According to this, first 1 component is very important
'''
    print [ "%.5f" % abs(i) for i in pca.components_[0] ]
'''
['0.00825', '0.00689', '0.68172', '0.51338', '0.05344', '0.00500', '0.00689', '0.40506', '0.00057', '0.29293', '0.07376', '0.02762', '0.11211', '0.00017']
# first component is has high abs values for 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options'
'''

def test_dt_importance():
    clf = tree.DecisionTreeClassifier(min_samples_split=5)
    importances = {}
    for _ in range(100):
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        clf.fit(features, labels)

        for i in range(len(clf.feature_importances_)) :
            key = str(features_list[i+1]) + ' ' + str(i+1)
            if key not in importances :
                importances[key] = 0

            importances[key] += clf.feature_importances_[i]


    # sort by importance
    importances = sorted(importances.items(), key=lambda x: x[1], reverse=True )

    pprint.pprint(importances)

    '''
 # after removing 'TOTAL' outlier
[('exercised_stock_options 10', 22.61587301587296),
 ('total_payments 3', 16.396472663139342),
 ('bonus 5', 12.691184549108078),
 ('restricted_stock 13', 12.428995637611505),
 ('expenses 9', 11.55927827692415),
 ('other 11', 6.7696506365557525),
 ('deferral_payments 2', 6.6073282895712682),
 ('long_term_incentive 12', 5.6634920634920629),
 ('total_stock_value 8', 2.177777777777778),
 ('deferred_income 7', 1.8201058201058213),
 ('salary 1', 1.2698412698412704),
 ('loan_advances 4', 0.0),
 ('director_fees 14', 0.0),
 ('restricted_stock_deferred 6', 0.0)]
     '''

    # find correlated fields between important features for task 3
    top_features_index = [10,3,5,13,9]
    text = ""
    for i in top_features_index:
        for j in top_features_index:
            text += "%.3f " % stats.pearsonr(features[i], features[j])[0]
        text += "\n"
    print text
'''
1.000 0.692 0.290 0.539 0.441
0.692 1.000 0.860 0.623 0.891
0.290 0.860 1.000 0.544 0.826
0.539 0.623 0.544 1.000 0.339
0.441 0.891 0.826 0.339 1.000
'''

    # use top 3 ['exercised_stock_options', 'total_payments', 'bonus' ]

if __name__ == '__main__':
    run()
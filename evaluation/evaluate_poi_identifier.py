#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import cPickle
import sys
from sklearn import cross_validation
from sklearn import tree
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = cPickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 



x_train, x_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print sum(pred), len(pred)


from sklearn.metrics import *
acc = accuracy_score(pred, y_test)

print precision_score(pred, y_test)
print recall_score(pred, y_test)

print acc

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

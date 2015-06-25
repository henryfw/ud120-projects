#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import cPickle
import sys
from sklearn import cross_validation
from sklearn import tree

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

out = open("../final_project/final_project_dataset.pkl", "r")

data_dict = cPickle.load(out)

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

x_train, x_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)
pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)

print acc

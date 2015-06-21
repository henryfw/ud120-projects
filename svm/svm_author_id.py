#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################


#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


from sklearn.svm import SVC

c_values = [10., 100., 1000., 10000.]
c_values = [10000.]

for c_value in c_values :
    clf = SVC(kernel="rbf", C=c_value)

    #### now your job is to fit the classifier
    #### using the training features/labels, and to
    #### make a set of predictions on the test data
    clf.fit(features_train, labels_train)

    #### store your predictions in a list named pred
    pred = clf.predict(features_test)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)

    print "C: %d\t\t%f" % (c_value, acc)

    print "ans: %d %d %d" % ( pred[10], pred[26], pred[50] )

    print "total chris %d " % sum(pred)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classifier_dict():
    classifier_dict = {
        'nb' : GaussianNB,
        'dt' : DecisionTreeClassifier,
        'svm' : SVC,
        'ada' : AdaBoostClassifier,
        'rf' : RandomForestClassifier
    }

    print "get_classifier_dict: ", [ i for i in classifier_dict ]
    return classifier_dict



def run():
    get_classifier_dict()





if __name__ == '__main__':
    run()
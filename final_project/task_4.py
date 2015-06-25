
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC



def get_classifier_dict():
    classifier_dict = {
        'nb' : GaussianNB(),
        'dt' : tree.DecisionTreeClassifier(min_samples_split=1),
        #'svm' : SVC()
    } 
    return classifier_dict



def run():
    get_classifier_dict()





if __name__ == '__main__':
    run()
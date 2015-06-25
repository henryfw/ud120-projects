import cPickle
import sys
import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import task_1, task_2, task_3_helper



# normalize all fields for SVM and adds features
def prepare_data(data_dict, fields = None, write_pickle = True) :

    if fields is None:
        fields = task_1.get_feature_list()

    data_dict = task_3_helper.add_new_feature(data_dict)
    data_dict = task_3_helper.normalize(data_dict, fields)


    # write to custom pkl file
    if write_pickle:
        with open('my_dataset.pkl', 'w') as f:
            cPickle.dump(data_dict, f)

    return data_dict

# new list with transformed data
def get_feature_list(feature_list = None, write_pickle = True) :
    if feature_list is None:
        feature_list = task_1.get_feature_list()

    new_feature_list = []
    for field in feature_list :
        if field == 'poi' :
            new_feature_list.append('poi')
        else:
            new_feature_list.append(field + '_n')

    print "task_4 get_feature_list: ", new_feature_list

    # write to custom pkl file
    if write_pickle:
        with open('my_feature_list.pkl', 'w') as f:
            cPickle.dump(new_feature_list, f)

    return new_feature_list



# from task 1 we see that expenses and total_payments are highly correlated. we can combine them into one feature
def run():
    features_list = get_feature_list(None, False)
    print features_list

    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = prepare_data( task_2.remove_outlier(data_dict), None, False)
    #pprint.pprint(  data_dict  )





if __name__ == '__main__':
    run()
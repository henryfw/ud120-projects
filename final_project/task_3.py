import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import task_1, task_2

# normalize all fields for SVM
def transform(data_dict) :

    expense_max = None
    expense_min = None
    tp_max = None
    tp_min = None

    for i in data_dict :
        expense = data_dict[i]['expenses']
        tp = data_dict[i]['total_payments']

        if expense != 'NaN' :
            if expense_max is None or expense > expense_max : expense_max = expense
            if expense_min is None or expense < expense_min : expense_min = expense
        if tp != 'NaN' :
            if tp_max is None or tp > tp_max : tp_max = tp
            if tp_min is None or tp < tp_min : tp_min = tp

    for i in data_dict :
        expense = ( data_dict[i]['expenses'] - expense_min ) / ( expense_max - expense_min ) if data_dict[i]['expenses'] != 'NaN' else 0
        tp = ( data_dict[i]['total_payments'] - tp_min ) / ( tp_max - tp_min ) if data_dict[i]['total_payments'] != 'NaN' else 0
        data_dict[i]['expenses_and_total_payments'] = expense + tp


    return data_dict


def normalize_data(data_dict) :
    fields_max = {}
    fields_min = {}
    fields = task_1.get_feature_list()

    for field in fields:
        if field == 'poi' :
            continue
        fields_max[field] = None
        fields_min[field] = None
        for i in data_dict :
            value = data_dict[i][field]
            if value != 'NaN' :
                if fields_max[field] is None or value > fields_max[field] : fields_max[field] = value
                if fields_min[field] is None or value < fields_min[field] : fields_min[field] = value

    for field in fields:
        if field == 'poi' :
            continue
        for i in data_dict :
            field_range = float(fields_max[field] - fields_min[field])
            if data_dict[i][field] == 'NaN':
                new_value = 0.
            else :
                new_value = data_dict[i][field]
                if field_range > 0 :
                    new_value = ( data_dict[i][field] - fields_min[field] ) / field_range
                #print field, data_dict[i][field], new_value, field_range, fields_max[field], fields_min[field]
            data_dict[i][field + '_n'] =  new_value

    return data_dict

# new list with transformed data
def get_feature_list() :
    feature_list = task_1.get_feature_list()
    new_feature_list = ['poi']
    for field in feature_list :
        if field == 'poi' :
            continue
        new_feature_list.append(field + '_n')
    return new_feature_list




# from task 1 we see that expenses and total_payments are highly correlated. we can combine them into one feature
def run():
    features_list = get_feature_list()
    print features_list

    data_dict = cPickle.load(open("final_project_dataset.pkl", "r") )
    data_dict = normalize_data( task_2.remove_outlier(data_dict))
    #pprint.pprint(  data_dict  )





if __name__ == '__main__':
    run()
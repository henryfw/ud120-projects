
import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import matplotlib.pyplot


my_dataset = cPickle.load(open("final_project_dataset.pkl", "r") )
features_list = ['poi', 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options' ]


def remove_outlier(data_dict):
    del data_dict['TOTAL']
    return data_dict


def run() :

    # plot shows an outlier for the "TOTAL"
    print_data(3, 0, 'task_2_with_outlier')


    # visually check without the TOTAL outlier
    print_data(3, 1, 'task_2_without_outlier')
    #print_data(1, 1, 'task_2_without_outlier')
    #print_data(2, 1, 'task_2_without_outlier')
    #print_data(3, 1, 'task_2_without_outlier')
    #print_data(4, 1, 'task_2_without_outlier')

    # they all look okay
    pass


# print the data with outlier removed
def print_data(column_index, number_to_remove, filename) :

    data = featureFormat(my_dataset, features_list, sort_keys = True)

    data = clean_data(data, column_index, number_to_remove)

    for i in range(len(data)):
        index = i + 1
        value = data[i][3]
        matplotlib.pyplot.scatter( index, value )

    matplotlib.pyplot.xlabel("index")
    matplotlib.pyplot.ylabel("value")
    matplotlib.pyplot.savefig(filename)


# removes extreme values, based on column_index
def clean_data(data, column_index, number_to_remove) :

    cleaned_data = []

    data_avg = sum([ item[column_index] for item in data ]) / len(data)

    # add the abs of difference in new col
    for i in range(len(data)):
        new_row = data[i].tolist()
        new_row.append(abs(data_avg - new_row[column_index]))
        cleaned_data.append(new_row)

    # sort by abs different between row value and mean of all row value
    cleaned_data = sorted(cleaned_data, key=lambda item: item[len(item) - 1], reverse=False)

    # return subset based on number_to_remove
    cleaned_data = cleaned_data[0 : len(cleaned_data) - number_to_remove ]

    return cleaned_data





if __name__ == '__main__':
    run()
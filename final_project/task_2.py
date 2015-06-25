


import cPickle
import sys
import pprint
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import matplotlib.pyplot


my_dataset = cPickle.load(open("final_project_dataset.pkl", "r") )
features_list = ['poi', 'expenses', 'exercised_stock_options', 'restricted_stock', 'total_stock_value' ]


def remove_outlier(data_dict):
    del data_dict['TOTAL']
    return data_dict



def run() :

    # plot shows an outlier for the "TOTAL"
    print_data(1, 0)


    # visually check without the TOTAL outlier
    print_data(1, 1)
    print_data(2, 1)
    print_data(3, 1)
    print_data(4, 1)

    # they all look okay


# print the data with outlier removed
def print_data(column_index, number_to_remove) :

    data = featureFormat(my_dataset, features_list, sort_keys = True)

    data = clean_data(data, column_index, number_to_remove)

    for i in range(len(data)):
        index = i + 1
        value = data[i][3]
        matplotlib.pyplot.scatter( index, value )

    matplotlib.pyplot.xlabel("index")
    matplotlib.pyplot.ylabel("value")
    matplotlib.pyplot.show()


# removes extreme values, based on column_index
def clean_data(data, column_index, number_to_remove) :

    cleaned_data = []

    data_avg = sum([ item[column_index] for item in data ]) / len(data)

    for i in range(len(data)):
        new_row = data[i].tolist()
        new_row.append(abs(data_avg - new_row[column_index]))
        cleaned_data.append(new_row)

    cleaned_data = sorted(cleaned_data, key=lambda item: item[len(item) - 1], reverse=False)
    cleaned_data = cleaned_data[0 : len(cleaned_data) - number_to_remove ]

    return cleaned_data





if __name__ == '__main__':
    run()
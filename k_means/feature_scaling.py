from sklearn.preprocessing import scale

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def get_rescaled(arr, i) :
    l = float(min(arr))
    m = float(max(arr))
    r = m - l
    return (float(i) - l) / r


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)



exercised_stock_options = []
salary = []
for i in data_dict:
    if data_dict[i]['exercised_stock_options'] != 'NaN':
        exercised_stock_options.append(float(data_dict[i]['exercised_stock_options']))
    if data_dict[i]['salary'] != 'NaN':
        salary.append(float(data_dict[i]['salary']))

print get_rescaled(salary, 200000)
print get_rescaled(exercised_stock_options, 1e6)
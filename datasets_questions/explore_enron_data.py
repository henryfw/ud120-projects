#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint
import sys

sys.path.append("../tools/")
from feature_format import featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print len(enron_data)

poi_total = 0
salaried_total = 0
has_email_total = 0

for i in enron_data:
    if enron_data[i]['poi']:
        poi_total += 1

    if enron_data[i]['salary'] != 'NaN':
        salaried_total += 1

    if enron_data[i]['email_address'] != 'NaN'  :
        has_email_total += 1


print salaried_total
print has_email_total


#np_data = featureFormat(enron_data, enron_data['SKILLING JEFFREY K'].keys() )
#print len( np_data["total_payments" == "NaN"] )

#pprint.pprint(enron_data["SKILLING JEFFREY K"])
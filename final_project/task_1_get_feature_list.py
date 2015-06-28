# help module to prevent cyclicaly imports in task_#

def get_feature_list() :

    # from dt importance: 'exercised_stock_options', 'total_payments', 'bonus'
    # items = ['poi', 'exercised_stock_options', 'total_payments', 'bonus' ]

    # from pca 1st component:  'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options'
    items = ['poi', 'total_payments', 'loan_advances', 'total_stock_value', 'exercised_stock_options', 'fraction_to_shared_with_poi' ]
    return items
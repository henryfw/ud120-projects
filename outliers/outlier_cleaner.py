#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """

    cleaned_data = []

    ### your code goes here
    for i in range(len(ages)):
        cleaned_data.append((
            ages[i], net_worths[i], abs(predictions[i] - net_worths[i])
        ))

    cleaned_data = sorted(cleaned_data, key=lambda item: item[2], reverse=False)
    cleaned_data = cleaned_data[0 : int(round(len(cleaned_data)*.9)) ]

    return cleaned_data

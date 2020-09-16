import numpy as np

from scipy.spatial.distance import cdist

from helper import error_report
from helper import seperate_data

def myKNN(training_data, test_data, k):
    """Run KNN on training and test data."""
    # Load data
    train_X, train_Y = seperate_data(training_data)
    test_X, test_Y = seperate_data(test_data)
    predictions = []
    for query in test_X:
        predictions += [knn(train_X, train_Y, query, k)]
    error = error_report(test_Y, predictions)
    print(f"Error for k = {k}: {error}")

def knn(train_X, train_Y, query, k):
    """Select prediction based on distances"""
    distances_indicies = []
    for i, feature in enumerate(train_X):
        distances_indicies += [(i, cdist(feature, query))]

    # Sort by distance and select smallest k values.
    sorted_distance = sorted(distances_indicies, key = lambda x: x[1])
    values = sorted_distance[:k]

    # Select class with most values.
    prediction_dict = {}
    for v in values:
        index = v[0]
        if train_Y[index] in prediction_dict:
            prediction_dict[train_Y[index]] += 1
        else:
            prediction_dict[train_Y[index]] = 1
    return max(prediction_dict, key = prediction_dict.get)

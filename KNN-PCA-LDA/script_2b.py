import numpy as np
import matplotlib.pyplot as plt

from helper import error_report
from helper import seperate_data
from helper import proportion_variance
from helper import to_dbl_lst

from myPCA import myPCA
from myKNN import myKNN

def main():
    """Problem 2b. Implement PCA on optdigits and run myKNN on the reduced dimensions"""
    # Load Data.
    train_data = np.loadtxt("optdigits_train.txt", delimiter = ',' , dtype = float)
    test_data = np.loadtxt("optdigits_test.txt", delimiter = ',', dtype = float)

    # Seperate Data into data and classification.
    train_X, train_Y = seperate_data(train_data)
    test_X, test_Y = seperate_data(test_data)

    # Convert classification to a list of lists.
    train_Y = to_dbl_lst(train_Y)
    test_Y = to_dbl_lst(test_Y)

    # Run myPCA on the training and testing data with num_principal_components = 21.
    train_principal_components, train_eigen_val = myPCA(train_data, 21, True)
    test_principal_components, test_eigen_val = myPCA(test_data, 21)

    # Append the classifications back onto the data.
    train_principal_components = np.append(train_principal_components, train_Y, axis = 1)
    test_principal_components = np.append(test_principal_components, test_Y, axis = 1)

    # Run myKNN on training and testing data.
    k_values = [1,3,5,7]
    for k in k_values:
        myKNN(train_principal_components, test_principal_components, k)

if __name__ == '__main__':
    main()

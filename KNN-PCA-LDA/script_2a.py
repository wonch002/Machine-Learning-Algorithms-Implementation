import numpy as np
import matplotlib.pyplot as plt

from helper import error_report
from helper import seperate_data
from myKNN import myKNN

def main():
    """Problem 2a. Implement KNN on optdigits"""
    train_data = np.loadtxt("optdigits_train.txt", delimiter = ',' , dtype = float)
    test_data = np.loadtxt("optdigits_test.txt", delimiter = ',', dtype = float)
    k_values = [1,3,5,7]
    for k in k_values:
        myKNN(train_data, test_data, k)

if __name__ == '__main__':
    main()

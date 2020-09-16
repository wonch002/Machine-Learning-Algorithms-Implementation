import numpy as np

from myLDA import myLDA
from myKNN import myKNN

def main():
    """Problem 2d. Implement LDA on optdigits and run myKNN on the reduced dimensions"""
    # Load Data.
    train_data = np.loadtxt("optdigits_train.txt", delimiter = ',' , dtype = float)
    test_data = np.loadtxt("optdigits_test.txt", delimiter = ',', dtype = float)
    # Run LDA on 3 values.
    l_values = [2,4,9]
    k_values = [1,3,5]
    for l in l_values:
        train_LDA, train_eigen_vals = myLDA(train_data, l)
        test_LDA, test_eigen_vals = myLDA(test_data, l)
        print(f"L = {l}")
        # Run myKNN on 3 values for each LDA.
        for k in k_values:
            myKNN(train_LDA, test_LDA, k)

if __name__ == '__main__':
    main()

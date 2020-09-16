import numpy as np
import matplotlib.pyplot as plt

from myPCA import myPCA
from helper import seperate_data
from helper import plot_PC

def main():
    """Problem 2c. Implement PCA on optdigigits. Project to 2-D space and visualize."""
    # Load Data.
    train_data = np.loadtxt("optdigits_train.txt", delimiter = ',' , dtype = float)
    test_data = np.loadtxt("optdigits_test.txt", delimiter = ',', dtype = float)

    # Seperate Data into data and classification.
    train_X, train_Y = seperate_data(train_data)
    test_X, test_Y = seperate_data(test_data)

    # Run myPCA on the training and testing data with num_principal_components = 2.
    train_principal_components, train_eigen_val = myPCA(train_data, 2)
    test_principal_components, test_eigen_val = myPCA(test_data, 2)

    # Plot Training and testing data.
    plot_PC(train_principal_components, train_Y)
    plot_PC(test_principal_components, test_Y)

if __name__ == '__main__':
    main()

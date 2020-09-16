import numpy as np
import matplotlib.pyplot as plt

from myLDA import myLDA
from helper import seperate_data
from helper import plot_PC

def main():
    """Problem 2e. Implement LDA on optdigigits. Project to 2-D space and visualize."""
    # Load Data.
    train_data = np.loadtxt("optdigits_train.txt", delimiter = ',' , dtype = float)
    test_data = np.loadtxt("optdigits_test.txt", delimiter = ',', dtype = float)

    # Seperate Data.
    train_X, train_Y = seperate_data(train_data)
    test_X, test_Y = seperate_data(test_data)

    # Run LDA on training and Testing.
    train_principal_components, train_eigen_val = myLDA(train_data, 2)
    test_principal_components, test_eigen_val = myLDA(test_data, 2)

    # Plot both Training and Testing.
    plot_PC(train_principal_components, train_Y)
    plot_PC(test_principal_components, test_Y)

if __name__ == '__main__':
    main()

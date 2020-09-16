import numpy as np
import matplotlib.pyplot as plt

from myPCA import myPCA
from helper import seperate_data

def main():
    """Problem 3a. Implement PCA on face data and visualize the first five faces."""
    train_data = np.loadtxt("face_test_data_960.txt", delimiter = ' ' , dtype = float)

    # Choosing 100 because that is what explains 90% of variance.
    train_principal_components, train_eigen_val = myPCA(train_data, 100)
    for i in range(5):
        plt.title(f"Face {i+1}")
        plt.imshow(np.reshape(train_principal_components[i], (10,10)))
        plt.show()

if __name__ == '__main__':
    main()

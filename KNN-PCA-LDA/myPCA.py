import numpy as np
import matplotlib.pyplot as plt

from helper import error_report
from helper import seperate_data
from helper import proportion_variance
from myKNN import myKNN

def myPCA(data, num_principal_components, proportion = False, back_project = False):
    """Principal Component Analysis"""
    # Split data.
    train_X, train_Y = seperate_data(data)

    # Calculate mean and covariance matrix
    mean = np.array((np.mean(train_X, axis = 0)).transpose())
    mean = np.squeeze(np.asarray(mean))
    covariance = np.cov(train_X, rowvar=0)

    # Compute Eigen Values from covariance matrix
    eigen_val, eigen_vec = np.linalg.eigh(covariance)

    # Combine Eigen Values and Vector Pairs and sort them.
    pairs = []
    for pair in zip(eigen_val, eigen_vec):
        pairs += [(pair)]
    pairs.sort(reverse = True, key=lambda pair:pair[0])
    eigen_val, eigen_vec = zip(*pairs)

    # Compute Proportion Variance and graph.
    if proportion:
        proportion_variance(eigen_val)

    # Project the eigen vectors to smaller space W^t(x-m).
    W = np.array(eigen_vec)
    scaled_data = np.array(train_X - mean).transpose()

    W = W[:num_principal_components]

    # Note: sig_eigen_vector is already transposed.
    principal_components = np.array(W.dot(scaled_data))

    # Only grab the significant values.
    significant_PC = (principal_components).transpose()
    significant_EV = eigen_val[:num_principal_components]
    # Check if we need to back_project.
    if not back_project:
        return (significant_PC, significant_EV)
    else:
        back_project_X = (W.transpose()).dot(significant_PC.transpose()).transpose() + mean
        return (back_project_X, significant_EV)

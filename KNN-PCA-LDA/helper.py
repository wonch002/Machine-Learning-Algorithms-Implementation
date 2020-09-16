import numpy as np
import matplotlib.pyplot as plt

def error_report(Y, predictions):
    """Report error rate of predictions vs actual values"""
    score = 0
    for actual, predict in zip(Y, predictions):
        if actual != predict:
            score += 1
    return score/Y.shape[0]

def seperate_data(dataset):
    """Split data into Features and Target."""
    X = []
    Y = []
    for i, data in enumerate(dataset):
        X += [dataset[i][:len(data)-1]]
        Y += [dataset[i][len(data) - 1]]
    X = np.matrix(X)
    Y = np.array(Y)
    return (X,Y)

def proportion_variance(eigen_val):
    num_eigen_vectors = []
    prop_of_variance = []
    min_eigen_vectors = -1

    for i, ev in enumerate(eigen_val):
        num_eigen_vectors += [i+1]
        prop_var = sum(eigen_val[:i+1])/sum(eigen_val)
        prop_of_variance += [prop_var]
        if min_eigen_vectors == -1 and prop_var >= 0.9:
            min_eigen_vectors = i+1

    # Plot and show proportion of Variance vs Num of Eigen Vectors.
    plt.scatter(num_eigen_vectors, prop_of_variance, c = 'c', marker = '+')
    plt.title('Proportion of Variance Explained')
    plt.axvline(min_eigen_vectors, color = 'r', linestyle = '--')
    plt.show()
    return min_eigen_vectors

def to_dbl_lst(lst):
    dbl_lst = []
    for i in lst:
        dbl_lst += [[i]]
    return np.array(dbl_lst)

def plot_PC(pc, Y):
    """Plot principal Components and their actual values"""
    x = []
    y = []
    colors = {
         0:'#DF0101' # Red
        ,1:'#DF7401' # Orange
        ,2:'#D7DF01' # Yellow
        ,3:'#74DF00' # Green
        ,4:'#01DF01' # Green darker
        ,5:'#01DF74' # Green Teal
        ,6:'#01DFD7' # Teal
        ,7:'#0174DF' # Blue
        ,8:'#0101DF' # Dark Blue
        ,9:'#7401DF' # Purple
    }
    for i, y in zip(pc, Y):
        plt.scatter(i[0], i[1], c = colors[y])

    plt.show()

import numpy as np
from helper import seperate_data

def myLDA(data, num_principal_components):
    # Split data.
    train_X, train_Y = seperate_data(data)

    # Intialize and compute the mean and covariance of each class.
    unique_classes = np.unique(train_Y)
    seperate_class = {}
    mean_dict = {}
    cov_dict = {}
    for c in unique_classes:
        temp = np.where(train_Y == c)
        temp_x = []
        for i in temp:
            temp_x += [train_X[i]]

        temp_x = np.array(temp_x)
        temp_x = np.reshape(temp_x, (temp_x.shape[2], temp_x.shape[1]))

        seperate_class[c] = temp_x

        mean = np.array((np.mean(temp_x, axis = 0)).transpose())
        mean = np.squeeze(np.asarray(mean))
        covariance = np.cov(temp_x, rowvar=1)

        mean_dict[c] = mean
        cov_dict[c] = covariance

    # Get the Eigen Vectors/Values
    eigen_dict = {}
    for c in cov_dict:
        eigen_val, eigen_vec = np.linalg.eigh(cov_dict[c])
        eigen_dict[c] = (eigen_val, eigen_vec)

    # Sort Eigen Vectors/Values.
    # Combine Eigen Values and Vector Pairs and sort them.
    for e in eigen_dict:
        pairs = []
        for pair in zip(eigen_dict[e][0], eigen_dict[e][1]):
            pairs += [(pair)]
        pairs.sort(reverse = True, key=lambda pair:pair[0])
        eigen_val, eigen_vec = zip(*pairs)
        eigen_dict[e] = (eigen_val, eigen_vec)

    total_data = np.empty((0,num_principal_components+1))
    eigen_values = []

    for c,e in zip(seperate_class, eigen_dict):
        # Grab first k eigen vectors.
        W = np.array(eigen_dict[e][1][:num_principal_components])
        W = W.transpose()
        X = seperate_class[c].transpose()
        lda = X.dot(W)
        lda = lda.reshape(lda.shape[0], num_principal_components)
        y = np.full((lda.shape[0],1), c)
        lda = np.append(lda, y, axis = 1)
        total_data = np.append(total_data, lda, axis = 0)
        eigen_values += [eigen_dict[e][0]]
    return (total_data, sum(eigen_values, ()))

from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal as mvn

def compress_image_KMeans(img):
    """Use KMeans (from sklearn) to compress the image to seven colors"""
    # Grab all colors
    colors = []
    for row in img:
        for col in row:
            colors.append(col)
    colors = np.array(colors)

    # Cluster all colors into seven colors using KMeans.
    kmeans = KMeans(n_clusters=7).fit(colors)
    labels = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create a new image from the clustered colors.
    new_img = np.zeros(shape=(img.shape))

    counter = 0
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            label = labels[counter]
            new_img[i][j] = means[label]
            counter += 1

    # Show the new image.
    plt.axis("off")
    plt.imshow(new_img)
    plt.show()

def calculate_loglikelihood(dataset, pi, means, covariances, num_clust, h):
    """Calculate the Log Likelihood for each distribution"""
    loglikelihood = 0
    for i in range(len(h)):
        ## Determine which cluster/distribution has the highest likelihood.
        k = np.where(h[i] == np.max(h[i]))
        k = np.sum(k)
        ## Calculate new log Likelihood
        loglikelihood += np.log(max(pi[k], 10e-5))
        loglikelihood += np.log(max(mvn.pdf(dataset[i], mean=means[k], cov=covariances[k]), 10e-5))
    return loglikelihood

def calculate_Q(dataset, pi, means, covariances, num_clust, h):
    """Calculate the Log Likelihood for each distribution"""
    loglikelihood = 0
    for i in range(len(h)):
        for j in range(len(h[i])):
            ## Calculate new commplete log Likelihood
            loglikelihood += h[i][j]*np.log(max(pi[j], 10e-5))
            loglikelihood += h[i][j]*np.log(max(mvn.pdf(dataset[i], mean=means[j], cov=covariances[j]), 10e-5))
    return loglikelihood

def e_step(dataset, pi, means, covariances, num_clust):
    """Perform the E-Step of EM"""

    # pi * multivariate guassian.
    ## pi(i) * mulitvariate guassian
    h = np.zeros(shape=(dataset.shape[0], num_clust))
    for d, data in enumerate(dataset):
        for k in range(num_clust):
            try:
                h[d][k] = pi[k]*mvn.pdf(dataset[d], mean=means[k], cov=covariances[k])
            except:
                raise TypeError("Error: Singular Covariance matrix detected.")

    # Normalize weights to 1
    ## Divide each by the sum of each -> pi(i)*multivariate guassian.
    for i, weight in enumerate(h):
        sum_weight = np.sum(weight)
        for j, w in enumerate(weight):
            h[i][j] = w/sum_weight

    # Return the Expectation Matrix for the E-Step.
    return h

def m_step(data, h, num_clust, flag):
    """Perform the M-Step of EM"""
    ## Determine how much weight each cluster should get. This is equal to pi -
    ## pi = SUM (h/N)
    ## Basically the prior - How likely is a point going to land in one of these
    ## Clusters. Should add up to 1. Equation 7.11 in textbook
    pi = np.sum(h, axis = 0)/h.shape[0]

    ## Calculate the Means. This is calculated by Summing h*x (where x is a data point).
    ## Next, divide this value by SUM(h). Remember: h is how likely our datapoint
    ## will be in a given cluster or distribution. Equation 7.13 in textbook.
    mean_dict = {}
    for k in range(num_clust):
        mean = np.zeros(data[0].shape)
        sum_h = 0
        for d, row in enumerate(data):
            for i in range(len(row)):
                mean[i] += row[i]*h[d][k]
            sum_h += h[d][k]
        mean_dict[k] = mean/sum_h

    ## Calculate the Covariance Matrix. This is calculated by taking SUM(h*(x-m)*(x-m)^t)
    ## The divide this new matrix by SUM(h). Equation 7.13 in textbook.
    cov_dict = {}
    for k in range(num_clust):
        covariance = np.zeros((mean_dict[0].shape[0], mean_dict[0].shape[0]))
        sum_h = 0
        for d, row in enumerate(data):
            ## Calculate (x-m)
            difference = row - mean_dict[k]
            difference = np.reshape(difference, (difference.shape[0], 1))
            ## Now calculate h * (x-m) * (x-m)^T
            calculation = h[d][k] * np.dot(difference, np.transpose(difference))

            ## If we have a singular matrix, account for this by adding a
            ## a regularization term.
            if flag:
                calculation += (0.01*np.identity(3))

            ## SUM over all data.
            covariance += calculation
            sum_h += h[d][k]

        if flag:
            while np.linalg.det(covariance/sum_h) == 0:
                covariance += .1
                print(covariance)
        cov_dict[k] = covariance/sum_h

    return (mean_dict, cov_dict, pi)

def EMG(img, k, flag):
    """Expectation Maximization Algorithm on Image Data"""
    # Grab all colors
    colors = []
    for row in img:
        for col in row:
            colors.append(col)
    colors = np.array(colors)

    # Generate random estimates.
    mean_dict = {}
    cov_dict = {}
    pi = {}

    ## Perform kMeans to calculate intial means.
    kmeans = KMeans(n_clusters = k, n_init = 3)
    kmeans.fit(colors)
    means = kmeans.cluster_centers_

    ## Intialize the covaraince matrix and pi.
    for i in range(k):
        mean_dict[i] = means[i]
        # Used the random library to do this. I hope thats ok.
        pi[i] = random.random()

        guess = random.randrange(0, colors.shape[0]-1)
        guess_two = random.randrange(0, colors.shape[0]-1)

        color_slice = colors[min(guess,guess_two):max(guess_two, guess)].transpose()

        # Ensure that covariance isn't singular.
        cov_dict[i] = np.cov(color_slice)
        while np.linalg.det(cov_dict[i]) == 0:
            cov_dict[i] += 0.1

    iterations = 101
    iteration = 0
    all_loglikelihoods = [0]
    complete_loglikelihood = []
    while iteration < iterations:

        # Compute how likely each piece of data belongs to a certain cluster.
        h = e_step(colors, pi, mean_dict, cov_dict, k)
        complete_loglikelihood += [calculate_Q(colors, pi, mean_dict, cov_dict, k, h)]
        # Compute a new mean vector and covariance matrix for each cluster.
        ## Additionally, compute a cluster_weight to determine how likely each
        ## cluster is.
        mean_dict, cov_dict, pi = m_step(colors, h, k, flag)
        complete_loglikelihood += [calculate_Q(colors, pi, mean_dict, cov_dict, k, h)]

        # Calculate a loglikelihood for the new parameters and track it.
        all_loglikelihoods += [calculate_loglikelihood(colors, pi, mean_dict, cov_dict, k, h)]
        converge = all_loglikelihoods[len(all_loglikelihoods)-1] - all_loglikelihoods[len(all_loglikelihoods)-2]
        print(converge)

        # Check exit condition.
        if abs(converge) < 10:
            ## Plot image...

            ## Create a new image from the clustered colors.
            new_img = np.zeros(shape=(img.shape))

            counter = 0
            for i, row in enumerate(img):
                for j, col in enumerate(row):
                    k_max = np.where(h[counter] == np.max(h[counter]))
                    k_max = np.sum(k_max)
                    new_img[i][j] = mean_dict[k_max]
                    counter += 1

            # Show the new image.
            plt.axis("off")
            plt.title(f"Stadium k = {k}")
            plt.imshow(new_img)
            plt.show()

            ## Plot likelihoods.
            M = []
            E = []
            for i, l in enumerate(complete_loglikelihood):
                if i%2 == 0: # Even
                    # M-step
                    M += [l]
                else:
                    # E-step
                    E += [l]
            plt.title(f"Graph k = {k}")
            plt.plot(np.linspace(0, len(M)-1, len(M)), M, 'xb')
            plt.plot(np.linspace(0, len(E)-1, len(E)), E, 'or')
            plt.legend(('M-Step', 'E-Step'))
            plt.ylabel('Log Likelihood')
            plt.show()
            return (h, mean_dict, complete_loglikelihood)

        iteration += 1

def main():
    # Load Stadium.
    stadium = io.imread('stadium.bmp',)
    stadium = stadium/255
    stadium = np.delete(stadium, 3, axis = 2)

    # Run EMG on stadium.
    k_values = [4, 8, 12]
    for k in k_values:
        EMG(stadium, k, False)

    # Load Goldy.
    goldy = io.imread('goldy.bmp')
    goldy = goldy/255
    goldy = np.delete(goldy, 3, axis = 2)

    # Run EMG on Goldy. Catch the error and display the error.
    try:
        EMG(goldy, 7, False)
    except:
        print("Error: Singular Matrix detected. Trying again with regularization term.")

    # Run EMG on Goldy with the flag as True (add a regularization term to covariance).
    EMG(goldy, 7, True)
    compress_image_KMeans(goldy)

if __name__ == '__main__':
    main()

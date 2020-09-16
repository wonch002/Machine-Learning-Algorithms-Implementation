import numpy as np

from helper import error_report
from helper import seperate_data

def MultiGaussian(training_data, testing_data, Model):
    # Load data
    train_data = np.loadtxt(training_data, delimiter = ',' , dtype = float)
    test_data = np.loadtxt(testing_data, delimiter = ',', dtype = float)
    train_X, train_Y = seperate_data(train_data)
    test_X, test_Y = seperate_data(test_data)

    # Calculate priors.
    # prior_one = percentage of Y = 1
    # prior_two = percentage of Y = 2
    prior_one = len(train_Y[np.where(train_Y==1)])/len(train_Y)
    prior_two = 1-prior_one

    # Split data set into two classes.
    class_one_X = np.empty(shape = [0,8])
    class_two_X = np.empty(shape= [0,8])
    for x, y in zip(train_X, train_Y):
        if y == 1:
            class_one_X = np.vstack((class_one_X, x))
        else:
            class_two_X = np.vstack((class_two_X, x))

    # Calculate mean and covariance matrix for each class.
    mean_one = (np.mean(class_one_X, axis = 0)).transpose()
    mean_two = (np.mean(class_two_X, axis = 0)).transpose()
    covariance_one = np.cov(class_one_X, rowvar=0)
    covariance_two = np.cov(class_two_X, rowvar=0)
    covariance = covariance_one*prior_one + covariance_two*prior_two

    if Model == 3:
        alpha_one = np.mean(covariance_one.diagonal())
        alpha_two = np.mean(covariance_two.diagonal())
        covariance_one = np.zeros(covariance_one.shape, float)
        covariance_two = np.zeros(covariance_two.shape, float)
        np.fill_diagonal(covariance_one, alpha_one)
        np.fill_diagonal(covariance_two, alpha_two)

    # Test model.
    predictions = []
    for feature in test_X:
        feature = feature.transpose()
        if Model == 1:
            class_one = model_one(feature, mean_one, covariance_one, prior_one)
            class_two = model_one(feature, mean_two, covariance_two, prior_two)
        elif Model == 2:
            class_one = model_two(feature, mean_one, covariance, prior_one)
            class_two = model_two(feature, mean_two, covariance, prior_two)
        elif Model == 3:
            class_one = model_two(feature, mean_one, covariance_one, prior_one)
            class_two = model_two(feature, mean_two, covariance_two, prior_two)
        else:
            raise Exception(f"Model {Model} not yet implemented!")
        if class_one >= class_two:
            predictions += [1]
        else:
            predictions += [2]

    print("--------------Error rate for Model {0}: {1}--------------".format(
        Model, error_report(test_Y, predictions)))
    print(f"\t* P(C1):{prior_one}")
    print(f"\t* P(C2):{prior_two}")
    print(f"\t* Mean_one:\n{mean_one}")
    print(f"\t* Mean_two:\n{mean_two}")
    if Model == 1 or Model == 2:
        print(f"\t* S1:\n{covariance_one}")
        print(f"\t* S2:\n{covariance_two}")
    if Model == 3:
        print("\t* alpha 1: {}".format(alpha_one))
        print("\t* alpha 2: {}".format(alpha_two))

def model_one(feature, mean, covariance, prior):
    """Calculation for model one"""
    first = (feature.shape[1]/2) * np.log(2*np.pi)

    second = 0.5 * np.log(np.linalg.det(covariance))

    third = 0.5 * ((feature - mean).transpose() * np.linalg.inv(covariance) * (feature - mean))

    fourth = np.log(prior)

    discriminant = -first - second - third + fourth
    return discriminant

def model_two(feature, mean, covariance, prior):
    """Calculation for model two"""
    first = 0.5 * ((feature - mean).transpose() * np.linalg.inv(covariance) * (feature - mean))

    second = np.log(prior)

    discriminant = -first + second
    return discriminant

def model_three(feature, mean, covariance, prior):
    """Calculation for model three"""
    first = np.log(2*np.pi)

    second = 0.5 * np.log(np.linalg.det(covariance))

    third = 0.5 * ((feature - mean).transpose() * np.linalg.inv(covariance) * (feature - mean))

    fourth = np.log(prior)

    discriminant = -first - second - third + fourth
    return discriminant

def main():
    print("--------------- Data Set 1 ---------------")
    MultiGaussian('training_data1.txt', 'test_data1.txt', 1)
    MultiGaussian('training_data1.txt', 'test_data1.txt', 2)
    MultiGaussian('training_data1.txt', 'test_data1.txt', 3)

    print("--------------- Data Set 2 ---------------")
    MultiGaussian('training_data2.txt', 'test_data2.txt', 1)
    MultiGaussian('training_data2.txt', 'test_data2.txt', 2)
    MultiGaussian('training_data2.txt', 'test_data2.txt', 3)

    print("--------------- Data Set 3 ---------------")
    MultiGaussian('training_data3.txt', 'test_data3.txt', 1)
    MultiGaussian('training_data3.txt', 'test_data3.txt', 2)
    MultiGaussian('training_data3.txt', 'test_data3.txt', 3)

if __name__ == '__main__':
    main()

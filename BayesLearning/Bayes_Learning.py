import numpy as np

# Import helper functions
from helper import seperate_data
from helper import error_report
from helper import bernoulli
from helper import prediction

def Bayes_Learning(training_data, validation_data):
    """Learn parameters and Tune prior function"""
    X,Y = seperate_data(training_data)

    # Means for each attribute sepearted by class.
    MLE_one = np.zeros(shape=(1,100))
    MLE_two = np.zeros(shape=(1,100))

    # Count of each class.
    count_one = 0
    count_two = 0

    # Fill up each list with a total count for each attribute.
    for x, y in zip(X,Y):
        if y == 1:
            count_one += 1
        else:
            count_two += 1
        for i, feature in enumerate(np.nditer(x)):
            if y == 1:
                MLE_one[0][i] += feature
            else:
                MLE_two[0][i] += feature

    # Calculate the mean for each attribute in each class.
    for i, feature in enumerate(np.nditer(MLE_one)):
        MLE_one[0][i] = MLE_one[0][i] / count_one
    for i, feature in enumerate(np.nditer(MLE_two)):
        MLE_two[0][i] = MLE_two[0][i] / count_two

    # Initilize variables for validation.
    X_valid , Y_valid = seperate_data(validation_data)
    predictions = []
    priors = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6]
    e = 2.71828182845904523536028747135
    error = []

    # Tune to the best prior.
    for p in priors:
        prior = 1-(e**-p)
        # Make predictions for dataset.
        predictions = prediction(X_valid, MLE_one, MLE_two, prior, 1-prior)
        # Report error of predictions.
        error += [error_report(Y_valid, predictions)]
        predictions = []

    # Report a table of priors and errors.
    best_prior = 0
    best_error = 101
    print("{:>6} | {:>6}".format("Prior", "Error"))
    print("-----------------")
    for p, a in zip(priors, error):
        # Table of errors and priors.
        if a < best_error:
            best_prior = p
            best_error = a
        print("{:>6} | {:>3}".format(p, a))
    print(f"Best sigma: {best_prior} || Error: {best_error} ")

    return (MLE_one, MLE_two, 1-(e**-best_prior))

def main():
    """Driver function"""
    training_data = np.loadtxt('training_data.txt', dtype = float)
    validation_data = np.loadtxt('validation_data.txt', dtype = float)
    Bayes_Learning(training_data, validation_data)

if __name__ == '__main__':
    main()

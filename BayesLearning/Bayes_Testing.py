import numpy as np

# Import helper functions
from helper import seperate_data
from helper import prediction
from helper import error_report
from Bayes_Learning import Bayes_Learning

def Bayes_Testing(test_data, p1, p2, pc1, pc2):
    """Test learned parameters and prior on Test Data."""
    X,Y = seperate_data(test_data)
    predictions = []
    # Make predictions
    predictions = prediction(X, p1, p2, pc1, pc2)
    print("Testing data Error Rate: {}".format(error_report(Y, predictions)))

def main():
    """Driver function"""
    training_data = np.loadtxt('training_data.txt', dtype = float)
    validation_data = np.loadtxt('validation_data.txt', dtype = float)
    p1, p2, prior = Bayes_Learning(training_data, validation_data)
    testing_data = np.loadtxt('testing_data.txt', dtype = float)
    Bayes_Testing(testing_data, p1, p2, prior, 1-prior)

if __name__ == '__main__':
    main()

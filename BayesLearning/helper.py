import numpy as np

def seperate_data(dataset):
    """Split data into Features and Target."""
    X = []
    Y = []
    for i, data in enumerate(dataset):
        X += [dataset[i][:100]]
        Y += [dataset[i][len(data) - 1]]
    X = np.matrix(X)
    Y = np.array(Y)
    return (X,Y)

def error_report(Y, predictions):
    """Report error rate of predictions vs actual values"""
    score = 0
    for actual, predict in zip(Y, predictions):
        if actual != predict:
            score += 1
    return score/Y.shape[0]

def bernoulli(MLE, feature):
    """Bernoulli function: (p^x)*(1-p)^(1-x) - Computes the LOG(Bernoulli)"""
    # likelihood = (MLE**feature) * ((1-MLE)**(1-feature))
    # log(likelihood) = (feature*log(MLE) + (1-feature)*log(1-MLE)
    likelihood = feature*np.log(MLE) + (1-feature)*np.log(1-MLE)
    if likelihood == 0:
        likelihood = 10**-10
    return likelihood

def prediction(X, p1, p2, pc1, pc2):
    """Make predictions for each row."""
    predictions = []
    for x in X:
        class_one = 1
        class_two = 1
        for i, feature in enumerate(np.nditer(x)):
            # Obtain MLE estimate for the features (Class 1 and 2).
            MLE_one = p1[0][i]
            MLE_two = p2[0][i]

            # Log Likelihood / Probability Density Function
            likelihood_one = bernoulli(MLE_one, feature)
            likelihood_two = bernoulli(MLE_two, feature)

            # Add all of the log likelihoods together.
            if feature == 0:
                class_one += likelihood_one
                class_two += likelihood_two
            else:
                class_one += 1-likelihood_one
                class_two += 1-likelihood_two

        # (prior * likelihood) is proportional to [(prior * likelihood)/evidence]
        # Discriminant Function...
        class_one_prediction = np.log(pc1) + class_one
        class_two_prediction = np.log(pc2) + class_two

        # Make a prediction by selecting the maximum prediction.
        if class_one_prediction >= class_two_prediction:
            predictions += [1]
        else:
            predictions += [2]

    return predictions

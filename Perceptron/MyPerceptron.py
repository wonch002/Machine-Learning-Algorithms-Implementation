import numpy as np
from scipy import io
from scipy.optimize import linprog
from matplotlib import pyplot as plt

def MyPerceptron(X, y, w0):
    error = 1
    N = len(X)
    iteration = 0
    limit = 1000
    while error > 0 and iteration <= limit:
        for t in range(N):
            # print (np.sign(np.inner(np.transpose(w0),X[t])), y[t])
            if np.sign(np.inner(w0, X[t])) != y[t]:
                # print("w0 is being recalculated...")
                w0 = w0 + y[t] * X[t]
                # print(f"w0 was set to {w0}")
        error = 0
        for j in range(N):
            if np.sign(np.inner(w0, X[j])) != y[j]:
                error += 1
        # print(f"There were {error} errors")
        iteration += 1
    return w0, iteration

def execute(data):
    """Execute everything with a given data-set"""
    # Initialize data.
    X = np.array(data['X'])
    Y = data['y']

    # Separate data.
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for row,value in zip(X,Y):
        # Classified as 1
        if value == 1:
            x1 += [row[0]]
            y1 += [row[1]]
        # Classified as -1
        else:
            x2 += [row[0]]
            y2 += [row[1]]

    # Initialize w to [1, -1]
    w = np.array([1,-1])

    # Set up graph.
    plt.scatter(x1, y1, c = 'b', marker = '*')
    plt.scatter(x2, y2, c = 'r', marker = '*')
    plt.title("Perceptron Algorithm Test")

    # Intialize x
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)

    # Calculate the slope and plot the original line.
    b = -w[1]/w[0] * x
    plt.plot(x, b, '--', c = 'g')

    # Run the perceptron algorithm.
    w, iteration = MyPerceptron(X,Y,w)
    if iteration < 1000:
        print(f"Perceptron found a solution in {iteration} iterations.")
    else:
        print(f"Limit of {iteration} reached - no solution was found.")

    # Calculate the slope and plot the new line - calculated MyPerceptron().
    b = -w[1]/w[0] * x
    plt.plot(x, b, c = 'g')

    plt.show()


def main():
    # Load data
    data1 = io.loadmat('data1.mat')
    data2 = io.loadmat('data2.mat')

    print("Performing Perceptron on data1...")
    execute(data1)

    print("Performing Perceptron on data2...")
    execute(data2)

main()

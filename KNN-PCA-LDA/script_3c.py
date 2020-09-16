import numpy as np
import matplotlib.pyplot as plt

from myPCA import myPCA

def main():
    """Problem 3c. Run PCA and then Backproject the dimensions to its original space. Visualize back Projection."""
    train_data = np.loadtxt("face_train_data_960.txt", delimiter = ' ' , dtype = float)

    k_values = [10,50,100]
    for k in k_values:
        back_projection, ev = myPCA(train_data, k, back_project = True)
        for i in range(5):
            plt.title(f"Back Projection; Face:{i+1}; K:{k}")
            plt.imshow(np.reshape(back_projection[i], (30,32)))
            plt.show()

if __name__ == '__main__':
    main()

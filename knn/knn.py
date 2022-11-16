from builtins import range
from builtins import object
import numpy as np
# from past.builtins import xrange


class knn(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass
      

    def train(self, X, y):
        """
        
        
        memorize the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) of the training data
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X, k = 1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_train, D) of the training data
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        
        dists = self.calc_distances(X)
        return self.predict_labels(dists, k = k)


    def calc_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        btrain = np.transpose(self.X_train) # 
        # s11 = a1*b1 + a2*b2 + ... an*bn 
        # s11 should be: (a1-b1)^2 + (a2-b2)^2 ...
        #                   = a1^2-2*a1*b1+b1^2 + ...
        #                     3x1   4x4    5x1
        b_test_train = -2 * np.matmul(X, btrain)          # 500x3072 X 3072x5000 = 500x5000   -2*a1*b1
        btrain_p2 = np.sum(np.power(btrain, 2), axis = 0) # sum over columns, 1x5000           a1^2
        b_test_train += btrain_p2                         # 500x5000 + 1x5000 = 500x5000       a1^2-2*a1*b1

        b_test_train = np.transpose(b_test_train)         # 5000x500 
        btest_p2 = np.sum(np.power(X, 2), axis = 1)       # sum over rows, 500x1               b1^2
        b_test_train += btest_p2                          # 5000x500 + 500x1 = 5000x500        a1^2-2*a1*b1*b1^2

        dists = np.sqrt(np.transpose(b_test_train))       #  500x5000

        return dists


    def predict_labels(self, dists, k = 1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0] # 
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            k_closest = np.argsort(dists[i])[0 : k] # take k nearest at column i 
            closest_y = self.y_train[k_closest] # row i, all columns. 

            d = {} # empty dictionary. map the repetitions:
            for c in closest_y:
              if c in d:
                d[c] += 1
              else:
                d[c] = 1
 

            allk = np.array(list(d.keys())); # extract the keys
            allv = np.array(list(d.values())); # extract the values
            idx = np.argsort(allv)              # sort the values in ascending order 
            maxk = allk[idx][-1]                # sort the keys according to the above order, take the last. 
             
            y_pred[i] = maxk 

        return y_pred


    def compute_L1_distance_2loops(self, X, matbias, pixbias):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the absolute distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
 
                """
                on Li norms:
                https://montjoile.medium.com/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c#:~:text=L1%20Norm%20is%20the%20sum,the%20vector%20are%20weighted%20equally.
                L0: the number of nonzero elements in the function.
                L1 (Manhattan): the sum of distances between points in the function.
                L2 (Euclidean): the shortest path between the points in the function. 
                """
                xtrain = self.X_train[j]    # training image j, 1x3072 
                xtest = X[i]                # test image i, 1x3072. 
                xdiff = np.subtract(xtrain, xtest) # vectors difference = vector, 1x3072
                xabs = np.absolute(xdiff)   # absolute of each pixel = vector 
                xsum = np.sum(xabs)         # sum pixels up = scalar per image

                dists[i, j] = xsum              

        return dists # matrix of distances between each training and each test point. 


 















































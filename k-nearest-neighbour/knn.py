import numpy as np 

class knn():
    def __init__(self, k):
        self.k=k

    def train(self, X, y):
        self.X_train  = X 
        self.y_train  = y 
    
    def predict(self,X_test, num_loops =2):
        if num_loops ==2:
            distances = self.compute_distance_two_loops(X_test)
        if num_loops ==1:
            distances = self.compute_distance_one_loops(X_test)
        if num_loops ==0:
            distances = self.compute_distances_no_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(distances)
    
    
    def compute_distances_no_loops(self, X_test):
        """
        Distance between each test point in X_test and each training point
        in self.X_train using no explicit loops.
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        
        # Equation (x-y)^2 = x^2 + y^2 - 2xy
        dists = np.reshape(np.sum(X_test**2, axis=1), [num_test,1]) + np.sum(self.X_train**2, axis=1) \
                - 2 * np.matmul(X_test, self.X_train.T)
        distances = np.sqrt(dists)
        return distances
    
    def compute_distance_one_loops(self, X_test):
        '''
        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data.
        
        Returns:
        - distances: A numpy array of shape (num_test, num_train) where distances[i]
        is the Euclidean distance between the ith test point and training point.   
        '''
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            distances[i,:] = np.sqrt(np.sum( (X_test[i,:] - self.X_train)**2, axis=1))
        return distances

    def compute_distance_two_loops(self, X_test):
        '''
        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data.
        
        Returns:
        - distances: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training
        point.    
        '''
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                distances[i,j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2))
        return distances

    def predict_labels(self, distances):
        num_test = distances.shape[0] 
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i,:])
            k_close_classes = self.y_train[y_indices[:self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_close_classes))
        
        return y_pred

if __name__ == "__main__":

    X = np.loadtxt('data.txt', delimiter = ',')
    y = np.loadtxt('targets.txt')
    
    knn = knn(k=3)
    knn.train(X,y)
    
    y_pred = knn.predict(X, num_loops=1)
    print(f'Accuracy: {sum(y_pred==y)/y.shape[0]}')
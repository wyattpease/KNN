import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k, distance_function):
        self.k = k
        self.distance_function = distance_function
        self.dist_tuple = None

    def train(self, features, labels):
        fl_tuple = []
        for i in range(len(features)):
            fl_tuple.append((features[i], float(labels[i])))

        self.dist_tuple = fl_tuple

    def get_k_neighbors(self, point):
        knn = []
        for x in self.dist_tuple:
            dist = self.distance_function(point, x[0])
            if self.k > len(knn):
                knn.append((dist, x[1]))
                knn.sort()
            else:
                if knn[self.k-1][0] > dist:
                    knn[self.k-1] = (dist, x[1])
                    knn.sort()
        d,n = zip(*knn)
        return n
   
    def predict(self, features):
        pred_labels = []
        for x in features:
            knn = self.get_k_neighbors(x)
            counter = 0
            for y in knn:
                if y == 0:
                    counter = counter - 1
                else:
                    counter = counter +1
            if (counter > 0):
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        return pred_labels

if __name__ == '__main__':
    print(np.__version__)

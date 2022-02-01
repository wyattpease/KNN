import numpy as np
from knn import KNN

def f1_score(real_labels, predicted_labels):
    assert len(real_labels) == len(predicted_labels)
    tp,tn,fp,fn = 0,0,0,0
    for x in range(len(real_labels)):
        if predicted_labels[x] == 1 and real_labels[x] == 1:
            tp = tp + 1
        elif predicted_labels[x] == 0 and real_labels[x] == 0:
            tn = tn+1
        elif predicted_labels[x] == 1 and real_labels[x] == 0:
            fp = fp+1
        elif predicted_labels[x] == 0 and real_labels[x] == 1:
            fn = fn + 1
            
    if fp == 0 and tp==0:
        return 0
    precision = float(tp/(tp+fp))
    recall = float(tp/(tp+fn))
    t = precision + recall
    if t == 0:
        return 0
    f1 = float(2 * precision * recall/t)
    return f1

class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        diff = np.subtract(point1, point2)
        diff_raised = np.power(np.abs(diff), 3)
        s = np.sum(diff_raised)
        return float(s ** (1/3))


    @staticmethod
    def euclidean_distance(point1, point2):
        sub = np.subtract(point1, point2)
        sq = np.square(sub)
        sum = np.sum(sq)

        return float(np.sqrt(sum))

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        dot = np.dot(point1, point2)
        sq = lambda t: t ** 2
        sq_func = np.vectorize(sq)
        p1_sq = sq_func(point1)
        p2_sq = sq_func(point2)
        sum = np.sum(p1_sq)
        sqrt = np.sqrt(sum)
        p1_norm = sqrt
        p2_norm = np.sqrt(np.sum(p2_sq))
        if (p1_norm == 0 or p2_norm == 0):
            return 1.00
        else:
            final = dot/(p1_norm * p2_norm)    
            return 1.00 - final      

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        best_f1 = 0
        for x in range(1, 30):
            if x % 2 != 0:           
                for y in distance_funcs:
                    knn = KNN(x, distance_funcs[y])
                    knn.train(x_train, y_train)
                    y_pred = knn.predict(x_val)
                    f1 = f1_score(y_val, y_pred)
                    if f1 > best_f1:
                        self.best_distance_function = y
                        self.best_k = x
                        self.best_model = knn
                        best_f1 = f1
                    elif f1 == best_f1:
                        if not self.best_distance_function == y:
                            if not self.best_distance_function == 'euclidean':
                                if y == 'minkowski':
                                    self.best_distance_function = y
                                    self.best_k = x
                                    self.best_model = knn
        
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        best_f1 = 0
        for x in range(1, 30):
            if x % 2 != 0:
                for z in scaling_classes:
                    for y in distance_funcs:
                        knn = KNN(x, distance_funcs[y])
                        scalar = scaling_classes[z]()
                        temp_train = np.copy(x_train)
                        temp_val = np.copy(x_val)
                        scaled_train = scalar(temp_train)
                        scaled_val = scalar(temp_val)
                        knn.train(scaled_train, y_train)
                        y_pred = knn.predict(scaled_val)
                        f1 = f1_score(y_val, y_pred)
                        if f1 > best_f1:
                            self.best_distance_function = y
                            self.best_k = x
                            self.best_model = knn
                            self.best_scaler = z
                            best_f1 = f1
                        elif f1 == best_f1:
                            if not self.best_scaler == z and z == 'min_max_scale':
                                self.best_scalar =  z
                                self.best_distance_function = y
                                self.best_k = x
                                self.best_model = knn
                            if self.best_scaler == z: 
                                if not self.best_distance_function == y:
                                    if not self.best_distance_function == 'euclidean':
                                        if not y == 'cosine_dist':
                                            self.best_scaler = z
                                            self.best_distance_function = y
                                            self.best_k = x
                                            self.best_model = knn


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        
        for f in range(len(features)):
            temp = features[f]
            sq = np.square(temp)
            sum = np.sum(sq)
            if sum != 0:
                sqrt = np.sqrt(sum)
                temp_div = np.divide(temp, sqrt)
                features[f] = temp_div
        return features


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        fl = np.asarray(features)
        f_t = fl.transpose()
        for f in range(len(f_t)):
            temp = f_t[f]
            max = np.max(temp)
            min = np.min(temp)
        
            if max != min:
                diff = max-min
                temp_sub = np.subtract(temp, min)
                temp_div = np.divide(temp_sub, diff)
                temp = temp_div
            else:
                temp = np.zeros(len(f_t[f]))
                
            f_t[f] = temp
        features = f_t.transpose()
        return features

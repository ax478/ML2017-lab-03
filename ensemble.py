import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''
    
    weak_classifier = None
    n_weaker_limit = None
    classifiers = []
    tree_weight = []
    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weaker_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = np.zeros(len(y))+float(1/len(y))
        for n in range(self.n_weaker_limit):
            self.weak_classifier.fit(X,y,sample_weight = w)
            y_pred = self.weak_classifier.predict(X)
            omega = 1 - np.sum(y_pred.T == y.T[0])/len(y)
            print(omega)
            alpha = 1/2*math.log((1-omega)/omega)
            self.classifiers.append(self.weak_classifier)
            self.tree_weight.append(alpha)
            w = w*np.exp(-alpha*y_pred.T*y.T[0])
            w = w/np.sum(w)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        score = np.zeros(X.shape[0])
        for i in range(self.n_weaker_limit):
            score += self.tree_weight[i]*self.classifiers[i].predict(X)
        return score

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        score = self.predict_scores(X)
        y_pred = np.sign(score)
        return y_pred
        


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

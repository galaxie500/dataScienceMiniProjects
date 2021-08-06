import numpy as np


class NaiveBayes_clf():
    
    def __init__(self, alpha=1, tf='f', tfidf=False):
        self.alpha = alpha
        self.tf = tf
        self.tfidf = tfidf
        print(f"Naive-Bayes-Multinomial-Classifier: alpha={self.alpha}, tf={self.tf}, tfidf={self.tfidf}")
        
    def fit(self, X, y):
        self.n_classes = np.unique(y).shape[0]
        self.n_docs = X.shape[0]
        self.n_words = X.shape[1]
        
        #vector(20,), number of documents in each class
        n_docs_class = np.array([y[y==j].shape[0] for j in range(1, self.n_classes+1)]) #class starts from 1
        self.prior = n_docs_class/np.sum(n_docs_class)
        
        self.likelihood = self._multinomials(X, y)
        
    def _multinomials(self, X, y):
        tf = np.zeros((n_classes, n_words))

        for j in range(20):
            jth_class = X[y==j+1]
            f = (np.sum(jth_class, axis=0) + self.alpha)/(np.sum(jth_class) + self.alpha*n_words)
            if self.tf == 'f':
                tf[j] = f
            else:
                tf[j] = np.log(1+f)
        return tf
    

    # calculate optimal solution for estimator
    def predict(self, X):
        if X_test.ndim == 1:
            n_test_samples = 1
        else:
            n_test_samples = X_test.shape[0]

        labels = np.zeros((n_test_samples, ))
        for i in range(n_test_samples):
            log_prob = np.sum(np.log(self.likelihood)*X[i], axis=1) + np.log(self.prior)
            labels[i] = np.argmax(log_prob)+1
        return labels
    
    def accuracy(self, X, y):
        acc = X[X==y].shape[0]/y.shape[0]
        print(f"Predicting on {y.shape[0]} samples, Accuracy:{round(acc, 3)}")
        return acc
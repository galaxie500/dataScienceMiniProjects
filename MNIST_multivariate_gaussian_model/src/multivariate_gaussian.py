class Gaussian_clf():
    
    def __init__(self, var_smoothing=1e-2):
        self.var_smoothing = var_smoothing
        
       
    def _prior_prob(self, y):
        self.n_classes = np.unique(y).shape[0]
        return np.array([y[y==i].shape[0]/y.shape[0] for i in range(self.n_classes)])
    
    def fit(self, X, y):
        """
        X: array-like of shape (n_samples, n_features)
           n_samples is the number of samples
           and n_features is the number of features.
        y: array-like of shape (n_classes,)
           n_classes is the number of classes
        """
        self.priors = self._prior_prob(y)
        self.n_features = X.shape[1]
        
        self.mu = np.array([X[np.where(y==i)].mean(axis=0) for i in range(self.n_classes)])
        
        sigma = np.array([np.cov(X[np.where(y==i)].T) for i in range(self.n_classes)])
        self.sigma = sigma + np.eye(self.n_features)*self.var_smoothing #smoothing covariance matrix
        
        return self
    
    
    def predict_proba(self, X):
        """
        X: array-like of shape (n_samples, n_features)
           
        probas: array-like of shape (n_samples, )
                maximum probabilty of samples among different classes
        """
        if X.ndim == 1:
            self.n_samples = 1
        else:
            self.n_samples = X.shape[0]
        
        #pass in the whole trainning data into mvn.logpdf()
        log_likelihood = np.array([mvn.logpdf(X, mean=self.mu[j],cov=self.sigma[j]) for j in range(self.n_classes)]) 
        # summation of log(p_j) and p_j(X) by broadcasting
        probas = np.log(self.priors) + log_likelihood.T
        
        return probas
    
        #using loops
        #probs = np.zeros((self.n_classes, self.n_samples))
        #for i in range(10):
            #probs[i] = np.log(pi[i]) + mvn.logpdf(X, mean=self.mu[i], cov=self.sigma[i])
        #return probs
    
        
    def predict(self, X):
        """
        X: array-like of shape (n_samples, n_features)
           
        res: array-like of shape (n_samples, )
            class labels for each sample
        """
        probs = self.predict_proba(X)
        
        return np.argmax(probs, axis=1)
    
    def get_param(self):
        return self.priors, self.mu, self.sigma
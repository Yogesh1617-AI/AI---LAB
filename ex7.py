from __future__ import annotations
import math
from typoing import Tuple, Dict, Any
import numpy as numpy

class NaiveBayesGaussian:

    def __init__(self, var_smoothing: float = le-9):
        self.var_smoothing = var_smoothing
        self.classes_:np.ndarray | None = None
        self.class_prior_log_:Dict[Any,float] = {}
        self.mean_:Dict[Any,np.ndarry] = {}
        self.var_:Dict[Any, np.ndarray] = {}

    def fit(self,X:np.ndarray)->"NaiveBayesGaussian":
        x = np.asarray(X,dtype = float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        for c in self.classes_:
            Xc = X[y == c]
            self.class_prior_log_[c] = math.log(len(Xc)/ len(X))
            mu = Xc.mean(axis = 0)
            var = Xc.var(axis = 0) + self.var_smoothing
            self.mean_[c] = mu
            self.var_[c] = var
        return self

    def _log_gaussian_likelihood(self, x:np.ndarray,c:Any)->float:
        mu = self.mean_[c]
        var = self.var_[c]
        return -0.5*(np.log(2*np.pi*var)).sum()-0.5*(((x - mu)**2)/var).sum()

    def _joint_log_likelihood(self, X:np.ndarray)->np.ndarray:
        scores = []
        for x in X:
            row = []
            for c in self.classes_:
                row.append(self.class_prior_log_[c] + self._log_gaussian_likelihood(x,c))
            scores.append(row)
        return np.array(scores)

    def predict(self,X:np.ndarray)->np.ndarray:
        x = np.asarray(X, dtype = float)
        jll = self._joint_log_likelihood(X)
        idx = jll.argmax(axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray)->np.ndarray:
        jll = self._joint_log_likelihood(np.asarray(X,dtype = float))
        jll -= jll.max(axis=1, keepdims = True)
        probs = np.exp(jll)
        probs /= probs.sum(axis=1, keepdims = True)
        return probs

class NaiveBayesMultinomial:

    def __init__(self, alpha:float = 1.0):
        if alpha<0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha
        self.classes_:np.ndarray | None


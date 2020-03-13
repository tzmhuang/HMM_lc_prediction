import numpy as np


class GaussianMixtureModel(object):
    def __init__(self, k=1, dim=1):
        self.dim = dim
        self.k = k
        return

    def initialize(self, c=None, mu=None, sigma=None):
        self.c = c if c != None else np.ones(self.k)*1/self.k
        self.mu = mu if mu != None else np.zeros((self.k, self.dim))
        self.sigma = sigma if sigma != None else np.array(
            [np.eye(self.dim)]*self.k)
        return

    def eval(self, obs):
        # NOTE: obs is 1d np array
        if len(obs) != self.dim:
            Exception("Observation dimetion error")
        prob = []
        for i in range(self.k):
            mu = self.mu[i]
            sigma = self.sigma[i]
            a = (((2*np.pi)**self.dim)*np.linalg.det(sigma))**0.5
            d = obs-mu
            b = -0.5*np.dot(np.dot(d, np.linalg.inv(sigma)), d)
            p = 1/a*np.exp(b)
            prob += [p]
        return np.dot(self.c, prob)

    def get_param(self, k):
        return self.c[k], self.mu[k], self.sigma[k]

    def set_param(self, k, c, mu, sigma):
        self.c[k] = c
        self.mu[k] = mu
        self.sigma[k] = sigma
        return

    def __repr__(self):
        s = "GMM with {0} Gaussian mixtures c: {1}| mu: {2}| sigma: {3}\n".format(
            self.k, self.c, self.mu, self.sigma)
        return s

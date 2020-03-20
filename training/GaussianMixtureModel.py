import numpy as np
eps = 10e-16


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
        # NOTE: obs must be nxd array
        if len(np.shape(obs)) > 1 and np.shape(obs)[1] != self.dim:
            raise ValueError("Observation dimesion error")
        elif len(np.shape(obs)) <= 1:
            raise ValueError("Data needs to be 2D array")
        print('*) EVAL_RESULT: ', [self.eval_k(obs, k)
                                   for k in range(self.k)])
        return np.sum([self.eval_k(obs, k) for k in range(self.k)], 0)

    def eval_k(self, obs, k):
        # NOTE: obs must be nxd array
        if len(np.shape(obs)) > 1 and np.shape(obs)[1] != self.dim:
            raise ValueError("Observation dimesion error")
        elif len(np.shape(obs)) <= 1:
            raise ValueError("Data needs to be 2D array")
        mu = self.mu[k]
        sigma = self.sigma[k]
        a = (((2*np.pi)**self.dim)*np.linalg.det(sigma))**0.5
        d = obs-mu
        b = -0.5*np.diag(d@np.linalg.inv(sigma)@d.T)
        p = 1/a*np.exp(b)+eps
        return self.c[k]*p

    def get_param(self):
        return self.c, self.mu, self.sigma

    def set_k_param(self, k, c, mu, sigma):
        self.c[k] = c
        self.mu[k] = mu
        self.sigma[k] = sigma
        return

    def set_param(self, c, mu, sigma):
        if c.shape == self.c.shape:
            self.c = c
        else:
            raise Exception('gmm c dimension incorrect')
        if mu.shape == self.mu.shape:
            self.mu = mu
        else:
            raise Exception('gmm mu dimension incorrect')
        if sigma.shape == self.sigma.shape:
            self.sigma = sigma
        else:
            raise Exception('gmm sigma dimension incorrect')
        return

    def __repr__(self):
        s = "GMM with {0} Gaussian mixtures c: {1}\n mu: {2}\n sigma: {3}\n".format(
            self.k, self.c, self.mu, self.sigma)
        return s

import numpy as np
from scipy import linalg


class GaussianMixtureModel(object):
    def __init__(self, n_mix=1, n_dim=1):
        self.n_dim = n_dim
        self.n_mix = n_mix
        return

    def initialize(self, c=None, mu=None, sigma=None):
        self.c = c if c is not None else np.ones(self.n_mix)*1/self.n_mix
        self.mu = mu if mu is not None else np.zeros(
            (self.n_mix, self.n_dim))
        self.sigma = sigma if sigma is not None else np.array(
            [np.eye(self.n_dim)]*self.n_mix)
        return

    def eval(self, obs):
        # NOTE: obs must be nxd array
        if len(np.shape(obs)) > 1 and np.shape(obs)[1] != self.n_dim:
            raise ValueError("Observation dimesion error")
        elif len(np.shape(obs)) <= 1:
            raise ValueError("Data needs to be 2D array")
        print('*) EVAL_RESULT: ', [self.eval_k(obs, k)
                                   for k in range(self.n_mix)])
        return np.sum([self.eval_k(obs, k) for k in range(self.n_mix)], 0)

    def eval_k(self, obs, k):
        # NOTE: obs must be nxd array
        if len(np.shape(obs)) > 1 and np.shape(obs)[1] != self.n_dim:
            raise ValueError("Observation dimesion error")
        elif len(np.shape(obs)) <= 1:
            raise ValueError("Data needs to be 2D array")
        mu = self.mu[k]
        sigma = self.sigma[k]
        a = (((2*np.pi)**self.n_dim)*np.linalg.det(sigma))**0.5
        d = obs-mu
        b = -0.5*np.diag(d@np.linalg.inv(sigma)@d.T)
        p = 1/a*np.exp(b)
        return self.c[k]*p

    def log_eval(self, obs, min_cov=10e-8):
        # returns:
        # (n_obs,n_mix) array
        # log probability density from gaussian mixes
        n_dim = self.n_dim
        n_mix = self.n_mix
        n_obs = len(obs)
        stat = zip(self.mu, self.sigma)
        log_p = np.zeros((n_obs, n_mix))
        for k, (mu, cov) in enumerate(stat):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                try:
                    cov_chol = linalg.cholesky(
                        cov+min_cov*np.eye(n_dim), lower=True)
                except linalg.LinAlgError:
                    raise ValueError(
                        "Matrix not positive definite or symmetric")
            log_det = 2*np.sum(np.log(np.diag(cov_chol)))
            sol = linalg.solve_triangular(
                cov_chol, (obs-mu).T, lower=True).T
            log_p[:, k] = -0.5*(n_dim*np.log(2*np.pi) +
                                log_det + np.sum(sol**2, axis=1))
        return log_p

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
            self.n_mix, self.c, self.mu, self.sigma)
        return s

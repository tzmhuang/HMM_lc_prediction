import numpy as np
from GaussianMixtureModel import GaussianMixtureModel


class HiddenMarkovModel(object):
    def __init__(self, states={0: 'default'}):
        self.states = states
        self.state_num = len(states)
        return

    def initialize(self, method='gmm', A=None, B=None, data_dim=1, gmm_k=1):
        self.method = method
        self.data_dim = data_dim
        if method == 'gmm':
            self.gmm_k = gmm_k
            self.A = np.ones(
                (self.state_num, self.state_num))*1/self.state_num
            self.B = []
            for i in self.states:
                gmm = GaussianMixtureModel(dim=data_dim, k=gmm_k)
                gmm.initialize()
                self.B += [gmm]

        else:
            raise Exception('No such initialization method')
        return

    def train(self, data):
        if self.method == 'gmm':
            # A_e: state transition matrix est
            emission_prob = self.emission_prob(data)
            alphas = self.alpha(emission_prob)
            betas = self.beta(emission_prob)
            A_e = np.eye(self.state_num)
            C_e = np.zeros((self.state_num, self.gmm_k))
            MU_e = np.zeros((self.state_num, self.gmm_k, self.data_dim))
            SIGMA_e = np.zeros(
                (self.state_num, self.gmm_k, self.data_dim, self.data_dim))
            for i in self.states:
                den = sum([self.xi(data, i, k, alphas, betas)
                           for k in self.states])
                for j in self.states:
                    A_e[i][j] = self.xi(data, i, j, alphas, betas)
                A_e[i] = A_e[i]/den
                c_est_den = np.sum([self.gmm_gamma(data, i, kk, alphas, betas)
                                    for kk in range(self.gmm_k)])
                for k in range(self.gmm_k):
                    gamma_sum = self.gmm_gamma(data, i, k, alphas, betas)
                    mu_est_num = self.gmm_gamma(
                        data, i, k, alphas, betas, mode='mu_est')
                    sigma_est_num = self.gmm_gamma(
                        data, i, k, alphas, betas, mode='sigma_est')
                    C_e[i][k] = gamma_sum
                    MU_e[i][k] = mu_est_num/gamma_sum
                    SIGMA_e[i][k] = sigma_est_num/gamma_sum
                C_e[i] = C_e[i]/c_est_den
            return A_e, C_e, MU_e, SIGMA_e
        else:
            raise Exception('No such train method')
        return

    def predict(self):
        pass

    def alpha(self, emission_prob):
        # TODO: Fix precision deteriation issue
        T = len(emission_prob)
        alphas = np.zeros(emission_prob.shape)
        alphas[0] = 1*emission_prob[0]  # setting state prior = 1
        for t in range(1, T):
            for s in self.states:
                alphas[t, s] = sum([alphas[t-1, i]*self.A[i, s]
                                    for i in self.states])*emission_prob[t, s]
        return alphas

    def beta(self, emission_prob):
        # TODO: Fix precision deteriation issue
        T = len(emission_prob)
        betas = np.zeros(emission_prob.shape)
        betas[T-1] = 1
        for t in range(T-2, -1, -1):
            betas[t] = np.sum(betas[t+1]*emission_prob[t+1]*self.A, 0)
        return betas

    def emission_prob(self, obs):
        T = len(obs)
        ep = np.array([self.B[s].eval(obs)
                       for s in self.states]).reshape(T, -1)
        return ep

    def fwbw_prob(self, obs, t):
        ep = self.emission_prob(obs)
        fp = self.alpha(ep)
        bp = self.beta(ep)
        return np.dot(fp[t], bp[t])

    def xi(self, obs, i, j, alphas, betas):
        # NOTE: Sum of Xi(i,j) for all t
        num = self.A[i][j]*np.dot(alphas[: -1, i]
                                  * self.B[j].eval(obs)[1:], betas[1:, j])
        den = np.dot(alphas[1], betas[1])
        return num/den

    def gmm_gamma(self, obs, j, k, alphas, betas, mode='sum'):
        # NOTE: Sum of Gamma(j, k) for all t
        num1 = alphas[:, j]*betas[:, j]
        den1 = np.dot(alphas[1], betas[1])
        num2 = self.B[j].eval_k(obs, k)
        den2 = self.B[j].eval(obs)
        if mode == 'sum':
            return np.dot(num1, num2/den2)/den1
        elif mode == 'mu_est':
            return np.dot(num1*(num2/den2), obs)/den1
        elif mode == 'sigma_est':
            d = obs - self.B[j].mu[k]
            sig = np.array([np.tensordot(o, o, 0) for o in obs])
            g = num1*(num2/den2)/den1
            return np.sum(np.array([g[i]*sig[i] for i in range(len(obs))]), 0)
        else:
            raise Exception('No such mode')

    def __repr__(self):
        return "Transition Matrix: %s\nEmission: %s" % (self.A, self.B)

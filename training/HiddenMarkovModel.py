import numpy as np
from GaussianMixtureModel import GaussianMixtureModel


class HiddenMarkovModel(object):
    def __init__(self, states={0: 'default'}):
        self.states = states
        self.state_num = len(states)
        return

    def initialize(self, method='gmm', A=None, B=None, dim=1):
        if method == 'gmm':
            self.A = np.ones(
                (self.state_num, self.state_num))*1/self.state_num
            self.B = []
            for i in range(self.state_num):
                gmm = GaussianMixtureModel(dim=dim, k=1)
                gmm.initialize()
                self.B += [gmm]

        else:
            Exception('No such initialization method')
        return

    def train(self, data, method='gmm'):
        if method == 'gmm':
            pass
        else:
            Exception('No such train method')
        pass

    def predict(self):
        pass

    def alpha(self, obs):
        T = len(obs)
        alphas = np.zeros((T, self.state_num))
        alphas[0] = np.array([gmm.eval(obs[0])
                              for gmm in self.B])
        for t in range(1, T):
            for n in range(self.state_num):
                alphas[t][n] = sum([
                    alphas[t-1][i]*self.B[n].eval(obs[t])*self.A[i][n] for i in range(self.state_num)])
        return alphas

    def beta(self, obs):
        T = len(obs)
        betas = np.zeros((T, self.state_num))
        betas[T-1] = 1
        for t in range(T-2, 0, -1):
            for n in range(self.state_num):
                betas[t][n] = sum(
                    [betas[t+1][j]*self.B[j].eval(obs[t+1])*self.A[n][j] for j in range(self.state_num)])
        return betas

    def __repr__(self):
        return "Transition Matrix: %s\nEmission: %s" % (self.A, self.B)

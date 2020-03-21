import os
import numpy as np
import pickle
import datetime
from GaussianMixtureModel import GaussianMixtureModel

eps = 10e-16
SAVE_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'saved_models/')


class HiddenMarkovModel(object):
    def __init__(self, states={0: 'default'}):
        self.states = states
        self.state_num = len(states)
        self.initialized = False
        return

    def initialize(self, method='gmm', data_dim=1, gmm_k=1):
        self.initialized = True
        self.method = method
        self.data_dim = data_dim
        if method == 'gmm':
            self.gmm_k = gmm_k
            self.A = np.ones(
                (self.state_num, self.state_num))*1/self.state_num
            # Random Initialization
            # self.A = np.random.rand(3, 3)
            # self.A = self.A/np.sum(self.A, 1).reshape(-1, 1)
            self.B = []
            for i in self.states:
                gmm = GaussianMixtureModel(dim=data_dim, k=gmm_k)
                gmm.initialize()
                self.B += [gmm]

        else:
            raise Exception('No such initialization method')
        return

    def train(self, data, epoch, thresh):
        if self.initialized == False:
            raise Exception('Model not initialized')
        if self.method == 'gmm':
            A_e = np.eye(self.state_num)
            C_e = np.zeros((self.state_num, self.gmm_k))
            MU_e = np.zeros(
                (self.state_num, self.gmm_k, self.data_dim))
            SIGMA_e = np.zeros(
                (self.state_num, self.gmm_k, self.data_dim, self.data_dim))
            break_flag = False
            for e in range(epoch):
                print("===> Epoch = {0} <===".format(e))
                # A_e: state transition matrix est
                emission_prob = self.emission_prob(data)
                # alphas = self.alpha(emission_prob)
                # betas = self.beta(emission_prob)
                alphas, nc = self.normalized_alpha(emission_prob)
                betas = self.normalized_beta(emission_prob, nc)
                print('--> EMISSION <--\n', emission_prob)
                print('--> alphas <--\n', alphas)
                print('--> betas <--\n', betas)
                print('--> Normalizing Constant <--\n', nc)
                for i in self.states:
                    den = sum([self.xi(data, i, k, alphas, betas)
                               for k in self.states])
                    print('--> {0} den: {1} <--\n'.format(i, den))
                    for j in self.states:
                        A_e[i][j] = self.xi(data, i, j, alphas, betas)
                    A_e[i] = A_e[i]/den
                    c_est_den = np.sum([self.gmm_gamma(data, i, kk, alphas, betas)
                                        for kk in range(self.gmm_k)])
                    for k in range(self.gmm_k):
                        gamma_sum = self.gmm_gamma(
                            data, i, k, alphas, betas)
                        mu_est_num = self.gmm_gamma(
                            data, i, k, alphas, betas, mode='mu_est')
                        sigma_est_num = self.gmm_gamma(
                            data, i, k, alphas, betas, mode='sigma_est')
                        C_e[i][k] = gamma_sum
                        MU_e[i][k] = mu_est_num/gamma_sum
                        SIGMA_e[i][k] = sigma_est_num / \
                            gamma_sum+np.eye(self.data_dim)*eps*10
                    C_e[i] = C_e[i]/c_est_den
                for s in self.states:
                    C, MU, SIGMA = self.B[s].get_param()
                    break_flag = break_flag and (np.abs(self.A-A_e[s]) < thresh).all() and (np.abs(C-C_e[s]) < thresh).all(
                    ) and (np.abs(MU-MU_e[s]) < thresh).all() and (np.abs(SIGMA-SIGMA_e[s]) < thresh).all()
                print('--> flag <--\n', break_flag)
                print('--> A_e <--\n', A_e)
                print('--> C_e <--\n', C_e)
                print('--> MU_e <--\n', MU_e)
                print('--> SIGMA_e <--\n', SIGMA_e)
                if break_flag:
                    return self.A, C, MU, SIGMA
                else:
                    self.update_param(
                        {'A': A_e, 'B': {'C': C_e, 'MU': MU_e, 'SIGMA': SIGMA_e}})
            return A_e, C_e, MU_e, SIGMA_e
        else:
            raise Exception('No such train method')
        return

    def predict(self):
        pass

    def update_param(self, param):
        if self.method == "gmm":
            self.A = param['A']
            B = param['B']
            for s in self.states:
                self.B[s].set_param(
                    B['C'][s], B['MU'][s], B['SIGMA'][s])
        else:
            pass
        return

    def emission_prob(self, obs):
        T = len(obs)
        ep = np.array([self.B[s].eval(obs)
                       for s in self.states]).T
        return ep

    def alpha(self, emission_prob):
        # TODO: Fix numerical underflow
        T = len(emission_prob)
        alphas = np.zeros(emission_prob.shape)
        alphas[0] = 1*emission_prob[0]  # setting state prior = 1
        for t in range(1, T):
            for s in self.states:
                alphas[t, s] = sum([alphas[t-1, i]*self.A[i, s]
                                    for i in self.states])*emission_prob[t, s]
        return alphas

    def normalized_alpha(self, emission_prob):
        # TODO: Fix numerical underflow
        T = len(emission_prob)
        alphas = np.zeros(emission_prob.shape)
        normalizing_constant = np.zeros(T)
        alphas[0] = 1*emission_prob[0]  # setting state prior = 1
        normalizing_constant[0] = np.sum(alphas[0])
        alphas[0] = alphas[0]/normalizing_constant[0]
        for t in range(1, T):
            for s in self.states:
                alphas[t, s] = sum([alphas[t-1, i]*self.A[i, s]
                                    for i in self.states])*emission_prob[t, s]
            normalizing_constant[t] = max(sum(alphas[t]), 0)
            alphas[t] = alphas[t]/normalizing_constant[t]
        return alphas, normalizing_constant

    def beta(self, emission_prob):
        # TODO: Fix numerical underflow
        T = len(emission_prob)
        betas = np.zeros(emission_prob.shape)
        betas[T-1] = 1
        for t in range(T-2, -1, -1):
            betas[t] = np.sum(betas[t+1]*emission_prob[t+1]*self.A, 1)
        return betas

    def normalized_beta(self, emission_prob, normalizing_constant):
        # TODO: Fix numerical underflow
        T = len(emission_prob)
        betas = np.zeros(emission_prob.shape)
        betas[T-1] = 1
        for t in range(T-2, -1, -1):
            betas[t] = np.sum(betas[t+1]*emission_prob[t+1]
                              * self.A, 1)/normalizing_constant[t]
        return betas

    def fwbw_prob(self, obs, t):
        ep = self.emission_prob(obs)
        fp = self.alpha(ep)
        bp = self.beta(ep)
        return np.dot(fp[t], bp[t])

    def xi(self, obs, i, j, alphas, betas):
        # NOTE: Sum of Xi(i,j) for all t
        print("INSIDE XI- [{0},{1}]".format(i, j))
        print(self.B[j])
        num = self.A[i][j]*np.dot(alphas[: -1, i]
                                  * self.B[j].eval(obs)[1:], betas[1:, j])
        den = np.dot(alphas[1], betas[1])
        return num/den

    def gmm_gamma(self, obs, j, k, alphas, betas, mode='sum'):
        # NOTE: Sum of Gamma(j, k) for all
        num1 = alphas[:, j]*betas[:, j]
        den1 = np.dot(alphas[1], betas[1])
        num2 = self.B[j].eval_k(obs, k)
        den2 = self.B[j].eval(obs)
        print("INSIDE GMM_GAMMA ({0}): \n".format(mode), den1, den2)
        if mode == 'sum':
            return np.dot(num1, num2/den2)/den1
        elif mode == 'mu_est':
            return np.dot(num1*(num2/den2), obs)/den1
        elif mode == 'sigma_est':
            div = obs - self.B[j].mu[k]
            sig = np.array([np.tensordot(d, d, 0) for d in div])
            g = num1*(num2/den2)/den1
            return np.sum(np.array([g[i]*sig[i] for i in range(len(obs))]), 0)
        else:
            raise Exception('No such mode')

    def save_model(self):
        if self.initialized == False:
            raise Exception('Model not initialzed')
        if self.method == 'gmm':
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            file_name = os.path.join(
                SAVE_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.model')
            if os.path.exists(file_name):
                raise Exception("File already exists")
            model = {}
            meta = {}
            meta['method'] = self.method
            meta['states'] = self.states
            meta['data_dim'] = self.data_dim
            meta['gmm_k'] = self.gmm_k
            param = {}
            param['A'] = self.A.tolist()
            C = []
            MU = []
            SIGMA = []
            for s in self.states:
                c, mu, sigma = self.B[s].get_param()
                C += [c.tolist()]
                MU += [mu.tolist()]
                SIGMA += [sigma.tolist()]
            param['B'] = {}
            param['B']['C'] = np.array(C)
            param['B']['MU'] = np.array(MU)
            param['B']['SIGMA'] = np.array(SIGMA)
            model['meta'] = meta
            model['param'] = param
            with open(file_name, 'wb') as f:
                pickle.dump(model, f)
        else:
            pass
        print('Model saved to: ', SAVE_DIR)
        return

    def load_model(self, load_dir):
        if self.initialized:
            raise Exception('Model already initialized')
        with open(load_dir, 'rb') as f:
            model = pickle.load(f)
        meta = model['meta']
        print(meta)
        if meta['method'] == 'gmm':
            self.states = meta['states']
            self.initialize(method= meta['method'], data_dim=meta['data_dim'], gmm_k=meta['gmm_k'])
            param = model['param']
            self.update_param(param)
        else:
            raise Exception('Unrecognized method')
        print('Model loaded from: ', load_dir)
        return

    def __repr__(self):
        if self.initialized:
            return "Transition Matrix: %s\nEmission: %s" % (self.A, self.B)
        else:
            return "HMM: Uninitialized"

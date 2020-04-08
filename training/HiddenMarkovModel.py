import os
import numpy as np
from sklearn import cluster
from sklearn.utils import check_array
import pickle
import datetime
from GaussianMixtureModel import GaussianMixtureModel

eps = 10e-16
SAVE_DIR = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'saved_models/')


class HiddenMarkovModel(object):
    def __init__(self, states={0: 'default'}, method='gmm',
                 data_dim=1, gmm_k=1, random_state=0, min_cov=1e-3, n_iter=20, thresh=10e-4):
        self.states = states
        self.initialized = False
        self.data_dim = data_dim
        self.random_state = random_state
        self.n_iter = n_iter
        self.thresh = thresh
        if method == 'gmm':
            self.method = method
            self.gmm_k = gmm_k
            self.min_cov = min_cov
            self.start_prob = np.ones(data_dim)/data_dim
        else:
            raise Exception('No such initialization method')
        return

    def _init_param(self, X):
        n_sts = len(self.states)
        n_mix = self.gmm_k
        n_dim = self.data_dim
        if self.method == 'gmm':
            self.A = np.ones((n_sts, n_sts))/n_sts
            self.B = []
            self.weights = np.zeros((n_sts, n_mix))
            self.means = np.zeros((n_sts, n_mix, n_dim))
            self.sigmas = np.zeros((n_sts, n_mix, n_dim, n_dim))
            initkmeans = cluster.KMeans(
                n_clusters=n_sts, random_state=self.random_state)
            g = initkmeans.fit_predict(X)
            cv = np.cov(X.T)+np.eye(n_dim)*self.min_cov
            for i in self.states:
                kmeans = cluster.KMeans(
                    n_clusters=n_mix, random_state=self.random_state)
                kmeans.fit(X[g == i])
                gmm = GaussianMixtureModel(n_dim=n_dim, n_mix=n_mix)
                sigma = np.zeros((n_mix, n_dim, n_dim))
                sigma[:] = cv
                weights = np.ones(n_mix)/n_mix
                gmm.initialize(
                    c=weights, mu=kmeans.cluster_centers_, sigma=sigma)
                self.weights[i] = weights
                self.means[i] = kmeans.cluster_centers_
                self.sigmas[i] = sigma
                self.B += [gmm]
            self.initialized = True
        else:
            return
        return

    def fit(self, X, n_iter):
        X = check_array(X)
        n_sts = len(self.states)
        n_obs, n_dim = X.shape
        n_mix = self.gmm_k
        if self.initialized is False:
            try:
                self._init_param(X)
            except:
                raise ValueError("Problem with initialization")
        if self.method == 'gmm':
            for i in range(n_iter):
                emitlogprob = self._log_emission(X)
                logA = np.log(self.A)
                fw = self._pass_forward(emitlogprob, logA)
                bw = self._pass_backward(emitlogprob, logA)
                # M-step update
                self.start_prob = np.exp(fw[0] + bw[0])
                self._normalize(self.start_prob, axis=0)
                # Update A
                logxisum = self._log_sum_xi(X, fw, bw, emitlogprob, logA)
                xi = np.exp(logxisum)
                new_A = xi.copy()
                self._normalize(new_A, axis=1)
                gamma = np.exp(self._log_gamma(X, fw, bw))
                gamma_sum = gamma.sum(axis=0)
                # Update Weights
                new_weights = gamma_sum.copy()
                self._normalize(new_weights, axis=1)
                # Update Mean
                means_num = np.einsum('jik,jh->ikh', gamma, X)
                means_den = gamma_sum[:, :, None]
                new_means = means_num/means_den
                # Update Sigma
                d = X[:, None, None, :] - self.means[None, :, :, :]
                sigmas_num = np.sum(
                    gamma[:, :, :, None, None]*(d[:, :, :, :, None]*d[:, :, :, None, :]), axis=0)
                sigmas_den = gamma_sum[:, :, None, None]
                new_sigmas = sigmas_num/sigmas_den
                self._update_param(
                    {'A': new_A, 'C': new_weights, 'MU': new_means, 'SIGMA': new_sigmas})
        else:
            raise ValueError(
                "\"{0}\" method not supported".format(self.method))

        return

    def predict(self):
        pass

    def _update_param(self, param):
        if self.method == "gmm":
            self.A = param['A'].copy()
            self.weights = param['C'].copy()
            self.means = param['MU'].copy()
            self.sigmas = param['SIGMA'].copy()
            for s in self.states:
                self.B[s].set_param(
                    param['C'][s].copy(), param['MU'][s].copy(), param['SIGMA'][s].copy())
        else:
            pass
        return

    def _log_emission(self, obs):
        n_obs, n_dim = obs.shape
        n_sts = self.state_num
        log_emission = np.zeros((n_obs, n_sts))
        for s in self.states:
            lge = self.B[s].log_eval(obs)
            lgc = np.log(self.B[s].c)
            log_emission[:, s] = self._lse(lge + lgc, axis=1)
        return log_emission

    def _pass_forward(self, emitlogprob, logA):
        n_obs, n_dim = emitlogprob.shape
        fwlogprob = np.zeros((n_obs, n_dim))
        temp = np.zeros(n_dim)
        for t in range(n_obs):
            if t == 0:
                fwlogprob[t] = emitlogprob[t] + np.log(self.start_prob)
            else:
                for s in range(n_dim):
                    for r in range(n_dim):
                        temp[r] = fwlogprob[t-1, r] + logA[r, s]
                    fwlogprob[t, s] = self._precision_lse(
                        temp) + emitlogprob[t, s]
        return fwlogprob

    def _pass_backward(self, emitlogprob, logA):
        n_obs, n_dim = emitlogprob.shape
        bwlogprob = np.zeros((n_obs, n_dim))
        for t in range(n_obs-2, -1, -1):
            bwlogprob[t] = self._precision_lse(
                bwlogprob[t+1]+emitlogprob[t+1]+logA, axis=1)
        return bwlogprob

    def _log_sum_xi(self, obs, fwlogprob, bwlogprob, emitlogprob, logA):
        n_obs, n_dim = obs.shape
        logsumxi = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                logsumxi[i, j] = self._lse(
                    fwlogprob[:n_obs-1, i] +
                    bwlogprob[1:, j]+emitlogprob[1:, j])+logA[i, j]
        den = self._lse(fwlogprob[-1])
        return logsumxi-den

    def _log_gamma(self, obs, fwlogprob, bwlogprob):
        n_obs, n_dim = obs.shape
        n_mix = self.gmm_k
        logprob = fwlogprob + bwlogprob
        loggamma = np.zeros((n_obs, n_dim, n_mix))
        den = self._lse(fwlogprob[0]+bwlogprob[0])
        for s in range(n_dim):
            logemit = self.B[s].log_eval(obs)+np.log(self.B[s].c)
            self._log_normalize(logemit, axis=1)
            loggamma[:, s, :] = logprob[:, s, np.newaxis] + logemit
        return loggamma - den

    def _lse(self, a, axis=None):
        return np.log(np.sum(np.exp(a), axis=axis))

    def _precision_lse(self, a, axis=None):
        temp = np.asarray(a, dtype=np.longdouble)
        max = np.max(a, axis=axis)
        return np.log(np.sum(np.exp(temp), axis=axis)/np.exp(max)) + max

    def _normalize(self, a, axis=None):
        sum = np.sum(a, axis=axis)
        if axis is not None:
            sum = np.asarray(sum)
            sum[sum == 0] = 1
            s = list(a.shape)
            s[axis] = 1
            sum.shape = s
        a /= sum

    def _log_normalize(self, a, axis=None):
        sum = self._lse(a, axis=axis)
        if axis > 0:
            s = list(a.shape)
            s[axis] = 1
            sum.shape = s
        a -= sum

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
            self.method = meta['method']
            self.data_dim = meta['data_dim']
            self.gmm_k = meta['gmm_k']
            self.A = np.zeros((self.data_dim, self.data_dim))
            self.B = []
            for s in self.states:
                gmm = GaussianMixtureModel(dim=self.data_dim, k=self.gmm_k)
                gmm.initialize()
                self.B += [gmm]
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

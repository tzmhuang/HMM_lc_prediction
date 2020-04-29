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
                 gmm_k=1, random_state=0, min_cov=1e-3, n_iter=20, thresh=10e-4):
        self.states = states
        self.initialized = False
        self.random_state = random_state
        self.n_iter = n_iter
        self.thresh = thresh
        self._convergence = []
        self._converged = False
        if method == 'gmm':
            self.method = method
            self.gmm_k = gmm_k
            self.min_cov = min_cov
        else:
            raise Exception('No such initialization method')
        return

    def _init_param(self, X):
        n_sts = len(self.states)
        n_mix = self.gmm_k
        n_obs, n_dim = X.shape
        if self.method == 'gmm':
            self.start_prob = np.ones(n_sts)/n_sts
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

    def fit(self, X, seq_length=[], n_iter=2):
        X = check_array(X)
        n_seq = max(len(seq_length), 1)
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
                start_prob_accum = np.zeros(n_sts)
                xisum_accum = np.zeros((n_sts, n_sts))
                gamma_sum_accum = np.zeros((n_sts, n_mix))
                means_num_accum = np.zeros((n_sts, n_mix, n_dim))
                sigmas_num_accum = np.zeros((n_sts, n_mix, n_dim, n_dim))
                logA = np.log(self.A)
                obs_logprob_accum = 0
                print('in loop 1', i)
                for j in range(n_seq):
                    print('in loop 2 ', j)
                    S = self._get_seq(X, seq_length, j)
                    emitlogprob = self._log_emission(S)
                    fw = self._pass_forward(emitlogprob, logA)
                    bw = self._pass_backward(emitlogprob, logA)
                    obs_logprob_accum += self._precision_lse(fw[0], axis=0)
                    print('fw: \n', fw)
                    print('bw: \n', bw)
                    print('framelogprob: \n', emitlogprob)
                    # accumulate start prob
                    temp = fw[0] + bw[0]
                    self._log_normalize(temp, axis=0)
                    start_prob_accum += np.exp(temp)
                    # accumulate logxisum
                    logxisum = self._log_sum_xi(
                        S, fw, bw, emitlogprob, logA)
                    print('logxisum: \n', logxisum)
                    xisum_accum += np.exp(logxisum)
                    # accumulate gamma_sum
                    gamma = np.exp(self._log_gamma(S, fw, bw))
                    print('gamma; \n', gamma)
                    gamma_sum_accum += gamma.sum(axis=0)
                    # accumulate means
                    means_num_accum += np.einsum('jik,jh->ikh', gamma, S)
                    # accumulate sigma
                    d = S[:, None, None, :] - self.means[None, :, :, :]
                    sigmas_num_accum += np.sum(
                        gamma[:, :, :, None, None]*(d[:, :, :, :, None]*d[:, :, :, None, :]), axis=0)
                # update start_prob
                prev_start_prob = self.start_prob.copy()
                self.start_prob = start_prob_accum
                self._normalize(self.start_prob, axis=0)
                print('update startprob: ', prev_start_prob,
                      '->', self.start_prob)
                # update A
                new_A = xisum_accum
                self._normalize(new_A, axis=1)
                # update weights
                new_weights = gamma_sum_accum.copy()
                print('new_weights num\n', new_weights)
                self._normalize(new_weights, axis=1)
                # update mean
                means_den = gamma_sum_accum[:, :, None]
                new_means = means_num_accum/means_den
                print('new_means num\n', means_num_accum)
                # update sigma
                sigmas_den = gamma_sum_accum[:, :, None, None]
                new_sigmas = sigmas_num_accum/sigmas_den
                print('new_covs num\n', sigmas_num_accum)
                # update parameter
                self._update_param(
                    {'A': new_A, 'C': new_weights, 'MU': new_means, 'SIGMA': new_sigmas})
                # update Convergence
                self._update_convergence(obs_logprob_accum)
                if self._converged:
                    print('Done')
                    print('Convergence: \n', self._convergence)
                    break
        else:
            raise ValueError(
                "\"{0}\" method not supported".format(self.method))

        return

    def predict(self):
        pass

    def _get_seq(self, X, seq_length, seq_id):
        pl = 0
        if len(seq_length) < 1:
            cl = len(X)
        else:
            cl = seq_length[seq_id]
        if seq_id > 0:
            pl = sum(seq_length[:seq_id])
            cl += pl
        return X[pl:cl]

    def _update_convergence(self, logprob):
        self._convergence.append(logprob)
        if len(self._convergence) < 2:
            return
        delta_logprob = self._convergence[-1] - self._convergence[-2]
        if delta_logprob < 0:
            self._converged = True
            print("Convergence Error: EM iteration diverged")
        else:
            if delta_logprob < self.thresh:
                self._converged = True
        return

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
        n_sts = len(self.states)
        log_emission = np.zeros((n_obs, n_sts))
        for s in self.states:
            lge = self.B[s].log_eval(obs)
            lgc = np.log(self.B[s].c)
            log_emission[:, s] = self._precision_lse(lge + lgc, axis=1)
        return log_emission

    def _pass_forward(self, emitlogprob, logA):
        n_obs, n_sts = emitlogprob.shape
        fwlogprob = np.zeros((n_obs, n_sts))
        temp = np.zeros(n_sts)
        for t in range(n_obs):
            if t == 0:
                fwlogprob[t] = emitlogprob[t] + np.log(self.start_prob)
            else:
                for s in range(n_sts):
                    for r in range(n_sts):
                        temp[r] = fwlogprob[t-1, r] + logA[r, s]
                    fwlogprob[t, s] = self._precision_lse(
                        temp) + emitlogprob[t, s]
        return fwlogprob

    def _pass_backward(self, emitlogprob, logA):
        n_obs, n_sts = emitlogprob.shape
        bwlogprob = np.zeros((n_obs, n_sts))
        for t in range(n_obs-2, -1, -1):
            bwlogprob[t] = self._precision_lse(
                bwlogprob[t+1]+emitlogprob[t+1]+logA, axis=1)
        return bwlogprob

    def _viterbi(self, emitlogprob, logA):
        n_obs, n_sts = emitlogprob.shape
        viterbi_prob = np.zeros((n_obs, n_sts))
        viberbi_bt = np.zeros((n_obs, n_sts))
        viterbi_path = np.zeros(n_obs)
        temp = np.zeros(n_sts)
        for t in range(n_obs):
            if t == 0:
                viterbi_prob[t] = emitlogprob[t]+np.log(self.start_prob)
            else:
                for s in range(n_sts):
                    temp = viterbi_prob[t-1]+logA[:, s]+emitlogprob[t]
                    viterbi_prob[t, s] = np.max(temp)
                    viberbi_bt[t, s] = np.argmax(temp)
        viterbi_score = np.max(viterbi_prob[-1])
        viterbi_path[-1] = np.argmax(viterbi_prob[-1])
        for t in range(n_obs-2, -1, -1):
            bt_idx = viterbi_path[t+1]
            viterbi_path[t] = viberbi_bt[t+1, int(bt_idx)]
        return (viterbi_score, viterbi_path)

    def _log_sum_xi(self, obs, fwlogprob, bwlogprob, emitlogprob, logA):
        n_sts = len(self.states)
        n_obs, n_dim = obs.shape
        logsumxi = np.zeros((n_sts, n_sts))
        for i in range(n_sts):
            for j in range(n_sts):
                logsumxi[i, j] = self._precision_lse(
                    fwlogprob[:n_obs-1, i] +
                    bwlogprob[1:, j]+emitlogprob[1:, j])+logA[i, j]
        den = self._precision_lse(fwlogprob[-1])
        return logsumxi-den

    def _log_gamma(self, obs, fwlogprob, bwlogprob):
        n_sts = len(self.states)
        n_obs, n_dim = obs.shape
        n_mix = self.gmm_k
        logprob = fwlogprob + bwlogprob
        loggamma = np.zeros((n_obs, n_sts, n_mix))
        den = self._precision_lse(fwlogprob[0]+bwlogprob[0])
        for s in range(n_sts):
            logemit = self.B[s].log_eval(obs)+np.log(self.B[s].c)
            self._log_normalize(logemit, axis=1)
            loggamma[:, s, :] = logprob[:, s, np.newaxis] + logemit
        return loggamma - den

    def _lse(self, a, axis=None):
        return np.log(np.sum(np.exp(a), axis=axis))

    def _precision_lse(self, a, axis=None):
        temp = np.asarray(a, dtype=np.longdouble)
        max = np.max(a, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(temp-max), axis=axis)) + max.squeeze(axis=axis)

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
        sum = self._precision_lse(a, axis=axis)
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
            meta['n_dim'] = self.means.shape[2]
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
            self.gmm_k = meta['gmm_k']
            self.A = np.zeros((len(meta['states']), len(meta['states'])))
            self.B = []
            for s in self.states:
                gmm = GaussianMixtureModel(dim=meta['n_dim'], k=self.gmm_k)
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

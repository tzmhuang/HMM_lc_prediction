import numpy as np
from HiddenMarkovModel import HiddenMarkovModel

print("**********Test Tnitialization **********")
# states = {0: 'sunny', 1: 'rain', 2: 'cloudy'}
states = {0: 'default', 1: 'other'}
hmm = HiddenMarkovModel(states, method='gmm', data_dim=2, gmm_k=1)
print(hmm)

print("**********Test Emission Porbability **********")
# obs = [[0, 0], [0, 0], [10, 10], [10, 10], [
#     0, 0], [10, 10], [10, 10], [0, 0], [0, 0]]

# obs= [[-1, -1], [-2, 2], [3, 3], [-1, -1], [0, 0], [-1, 1], [3, 3], [-2, 2], [1, 0]]

obs = [[1, 1], [1, 1], [1.1, 1], [0.9, 1], [
    1, 1], [1, 1], [1.2, 1], [0.8, 1], [1, 1]]
# np.random.seed(0)
# obs = np.ones((100000, 2))+np.random.rand(100000, 2)
print(obs)

hmm._init_param(np.array(obs))
print(hmm.weights)
print(hmm.means)
print(hmm.sigmas)


emitlogprob = hmm._log_emission(np.array(obs))
print(emitlogprob)

print("**********Test Foward Porbability **********")
logA = np.log(hmm.A)
a = hmm._pass_forward(emitlogprob, logA)
print(a)
print('fw likelihood: ', hmm._lse(a[-1]))

print("**********Test Backward Porbability **********")

b = hmm._pass_backward(emitlogprob, logA)
print(b)
print('fwbw likelihood: ', hmm._lse(a + b, 1))

print("**********Test Gamma Calculation **********")
loggamma = hmm._log_gamma(np.array(obs), a, b)
print('loggamma:\n', np.exp(loggamma))
print(loggamma.sum(0).shape)

print("**********Test Xi Calculation **********")
xi = hmm._log_sum_xi(np.array(obs), a, b, emitlogprob, logA)
print(xi)

print("**********Test Training Process **********")
hmm.fit(obs, 2)
print('\n=====> RESULT <=====')
print('Transision Mat:\n', hmm.A)
print('C: \n', hmm.weights)
print('MEAN: \n', hmm.means)
print('SIGMA: \n', hmm.sigmas)


print("**********Test Save Model **********")
hmm.save_model()


print("**********Test Load Model **********")
load_hmm = HiddenMarkovModel()
dir = './saved_models/20200321_233110.model'
load_hmm.load_model(dir)

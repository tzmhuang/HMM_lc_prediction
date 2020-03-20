import numpy as np
from HiddenMarkovModel import HiddenMarkovModel

# states = {0: 'sunny', 1: 'rain', 2: 'cloudy'}
states = {0: 'default', 1: 'other'}
hmm = HiddenMarkovModel(states)

print("**********Test Tnitialization **********")
hmm.initialize(data_dim=2, gmm_k=1)
print(hmm.A, hmm.B)
print(hmm)

print("**********Test Emission Porbability **********")
obs = [[-1, 1], [2, -2], [-3, 3], [0, 0], [
    0, 0], [1, 0], [0, 0], [-1, 1], [3, -3]]

# obs = [[1, 1], [1, 1], [1.1, 1], [0.9, 1], [
#     1, 1], [1, 1], [1.2, 1], [0.8, 1], [1, 1]]

emission_prob = hmm.emission_prob(obs)
print(emission_prob)
print("**********Test Foward Porbability **********")

a = hmm.alpha(emission_prob)
print(a)
print(sum(a[-1]))

print("**********Test Backward Porbability **********")
b = hmm.beta(emission_prob)
print(b)
print(np.dot(np.array([hmm.B[i].eval(obs)
                       for i in states]).reshape(-1, len(states))[1], b[0]))
print(hmm.fwbw_prob(obs, 1))

print("**********Test Gamma Calculation **********")
print('Mode Sum: ', hmm.gmm_gamma(obs, 0, 0, a, b))
print('Mode: Mu_est: ', hmm.gmm_gamma(obs, 0, 0, a, b, mode='mu_est'))
print('Mode: Sigma_est: ', hmm.gmm_gamma(
    obs, 0, 0, a, b, mode='sigma_est'))

print("**********Test Xi Calculation **********")
print('0 to 0: ', hmm.xi(obs, 0, 0, a, b))
print('0 to 1: ', hmm.xi(obs, 0, 1, a, b))
# print('0 to 2: ', hmm.xi(obs, 0, 2, a, b))
print('1 to 0: ', hmm.xi(obs, 1, 0, a, b))
print('1 to 1: ', hmm.xi(obs, 1, 1, a, b))
# print('1 to 2: ', hmm.xi(obs, 1, 2, a, b))
# print('2 to 0: ', hmm.xi(obs, 2, 0, a, b))
# print('2 to 1: ', hmm.xi(obs, 2, 1, a, b))
# print('2 to 2: ', hmm.xi(obs, 2, 2, a, b))

print("**********Test Training Process **********")
A, C, MU, SIGMA = hmm.train(obs, 1000, 10e-3)
print('\n=====> RESULT <=====')
print('Transision Mat:\n', A)
print('MEAN: \n', MU)
print('C: \n', C)
print('SIGMA: \n', SIGMA)

emission_prob = hmm.emission_prob(obs)
print('EMISSION: \n', emission_prob)
a = hmm.alpha(emission_prob)
print(a)
print(sum(a[-1]))

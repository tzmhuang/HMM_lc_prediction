import numpy as np
from HiddenMarkovModel import HiddenMarkovModel

states = {0: 'sunny', 1: 'rain', 2: 'cloudy'}
hmm = HiddenMarkovModel(states)
hmm.initialize(dim=1)


print(hmm.A, hmm.B)
print(hmm)

obs = [[0], [0]]
a = hmm.alpha(obs)
print(a)
print(sum(a[-1]))

b = hmm.beta(obs)
print(b)

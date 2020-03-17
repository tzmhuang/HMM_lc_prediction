import numpy as np
from GaussianMixtureModel import GaussianMixtureModel

gmm = GaussianMixtureModel(dim=2, k=3)
gmm.initialize()

print(gmm.c)
print(gmm.mu)
print(gmm.sigma)


c, mu, sigma = gmm.get_param(1)
print(c, mu, sigma)
print(gmm)

obs = [[0, 0], [0, 0], [0, 0], [0, 0], [
    0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

print(gmm.eval_k(obs, 1))
print(gmm.eval(obs))

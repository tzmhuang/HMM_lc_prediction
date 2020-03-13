import numpy as np
from GaussianMixtureModel import GaussianMixtureModel

gmm = GaussianMixtureModel(dim=3, k=2)
gmm.initialize()

print(gmm.c)
print(gmm.mu)
print(gmm.sigma)


c, mu, sigma = gmm.get_param(1)
print(c, mu, sigma)

print(gmm)

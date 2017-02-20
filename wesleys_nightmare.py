"""

Wesley's nightmare

"""

from strategy import *
from bandit import *


# Bernoulli
bernoulli_bandit = StaticBernoulli(probs=[0.3, 0.7])
bernoulli_bandit.pull_arm(1)

my_strategy = ThompsonBernoulli(bandit=bernoulli_bandit,
                                alpha_prior=1,
                                beta_prior=1)
my_strategy.fit(iterations=10000)
print(my_strategy.alpha_posterior)
print(my_strategy.beta_posterior)

print(1.0 * my_strategy.alpha_posterior[0] / (my_strategy.alpha_posterior[0] + my_strategy.beta_posterior[0]))
print(1.0 * my_strategy.alpha_posterior[1] / (my_strategy.alpha_posterior[1] + my_strategy.beta_posterior[1]))


# Gaussian
gaussian_bandit = StaticGaussian(mu=[100, 200], sigma=[10, 10])
gaussian_bandit.pull_arm(1)

my_strategy = ThompsonGaussianKnownSigma(bandit=gaussian_bandit,
                                         sigma=10, mu_prior=0, sigma_prior=100)
my_strategy.fit(iterations=10000)
print(my_strategy.mu_posterior)
print(my_strategy.sigma_posterior)


"""

Thompson sampling example

"""

import numpy as np

from arm import BernoulliArm, GaussianArm
from bandit import StaticBandit, LinearInterpolationBandit
from strategy import ThompsonBernoulli, ThompsonGaussianKnownSigma

if __name__ == '__main__':
    np.random.seed(0)

    # Bernoulli
    bernoulli_bandit = StaticBandit(arms=[BernoulliArm(prob=0.45),
                                          BernoulliArm(prob=0.55)])
    bernoulli_strategy = ThompsonBernoulli(bandit=bernoulli_bandit,
                                           alpha_prior=1, beta_prior=1)
    result = bernoulli_strategy.fit(iterations=1000)
    print('Prior params: ' + str(bernoulli_strategy.prior_params))
    print('Posterior params: ' + str(bernoulli_strategy.posterior_params))
    print('True means: ' + str([0.45, 0.55]))
    print('Mean reward estimates: ' + str(bernoulli_strategy.mean_reward_estimates))

    # Gaussian
    gaussian_bandit = StaticBandit(arms=[GaussianArm(mu=95, sigma=30),
                                         GaussianArm(mu=105, sigma=30)])
    gaussian_strategy = ThompsonGaussianKnownSigma(bandit=gaussian_bandit,
                                                   sigma=20,
                                                   mu_prior=0, sigma_prior=200)
    gaussian_strategy.fit(iterations=500)
    print('Prior params: ' + str(gaussian_strategy.prior_params))
    print('Posterior params: ' + str(gaussian_strategy.posterior_params))
    print('True means: ' + str([95, 105]))
    print('Mean reward estimates: ' + str(gaussian_strategy.mean_reward_estimates))

    # Linear interpolation bandit with Gaussian errors
    dynamic_bandit = LinearInterpolationBandit(means=np.array([[5.0, 8.0], [10.0, 5.0]]),
                                               periods=[200, 200])
    dynamic_strategy = ThompsonGaussianKnownSigma(bandit=dynamic_bandit,
                                                  sigma=20,
                                                  mu_prior=0, sigma_prior=200,
                                                  memory_multiplier=0.9)
    dynamic_strategy.fit(iterations=1000, plot=True)
    print('Prior params: ' + str(dynamic_strategy.prior_params))
    print('Posterior params: ' + str(dynamic_strategy.posterior_params))


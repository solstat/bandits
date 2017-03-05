"""

Thompson sampling example

"""

import numpy as np

from arm import BernoulliArm, GaussianArm
from bandit import StaticBandit
from strategy import ThompsonBernoulli, ThompsonGaussianKnownSigma

if __name__ == '__main__':
    np.random.seed(0)

    # Bernoulli
    bernoulli_bandit = StaticBandit(arms=[BernoulliArm(prob=0.45),
                                          BernoulliArm(prob=0.55)])
    thompson_bernoulli = ThompsonBernoulli(bandit=bernoulli_bandit,
                                           alpha_prior=1, beta_prior=1)
    result = thompson_bernoulli.fit(iterations=1000)
    print('Prior params: ' + str(thompson_bernoulli.prior_params))
    print('Posterior params: ' + str(thompson_bernoulli.posterior_params))
    print('True means: ' + str([0.45, 0.55]))
    print('Mean reward estimates: ' + str(thompson_bernoulli.mean_reward_estimates))

    # Gaussian
    gaussian_bandit = StaticBandit(arms=[GaussianArm(mu=95, sigma=30),
                                         GaussianArm(mu=105, sigma=30)])
    gaussian_bandit.pull_arm(1)

    thompson_gaussian = ThompsonGaussianKnownSigma(bandit=gaussian_bandit,
                                                   sigma=20,
                                                   mu_prior=0, sigma_prior=200)
    thompson_gaussian.fit(iterations=500)
    print('Prior params: ' + str(thompson_gaussian.prior_params))
    print('Posterior params: ' + str(thompson_gaussian.posterior_params))
    print('True means: ' + str([95, 105]))
    print('Mean reward estimates: ' + str(thompson_gaussian.mean_reward_estimates))


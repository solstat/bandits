"""

ThompsonSampling Strategy

"""

import numpy as np
import scipy.stats
from copy import deepcopy
from strategy.strategy import Strategy


class ThompsonSampling(Strategy):
    """ Thompson Sampling Strategy

    Args:
        bandit (Bandit): bandit
        **kwargs : prior parameters

    Attributes:
        num_arms (int)
        posterior_params (list of list): posterior parameters
        estimated_arm_means (ndarray): posterior predictive mean
        estimated_arm_sds (ndarray): posterior predictive standard deviation
    """
    def __init__(self, bandit, **kwargs):
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.prior_params = [deepcopy(kwargs) for _ in range(self.num_arms)]
        self.posterior_params = deepcopy(self.prior_params)

    def fit(self, iterations, restart=False):
        if restart == True:
            self._restart()

        index_arms_pulled = [None] * iterations
        observed_rewards = [None] * iterations
        estimated_arm_means = [None] * iterations
        estimated_arm_sds = [None] * iterations
        for i in range(iterations):
            index_arms_pulled[i] = self._choose_arm()
            observed_rewards[i] = self.pull_arm(index_arms_pulled[i])
            self._update_posterior(index_arms_pulled[i], observed_rewards[i])
            estimated_arm_means[i] = self.estimated_arm_means
            estimated_arm_sds[i] = self.estimated_arm_sds

        out = {
            'rewards': np.array(observed_rewards),
            'arms_pulled': np.array(index_arms_pulled),
            'estimated_arm_means': np.array(estimated_arm_means),
            'estimated_arm_sds': np.array(estimated_arm_sds)
        }
        return out

    def _restart(self):
        self.posterior_params = deepcopy(self.prior_params)

    def _choose_arm(self):
        samples = [self._sample(**params) for params in self.posterior_params]
        return np.argmax(samples)

    def pull_arm(self, arm_index):
        return self.bandit.pull_arm(arm_index)

    def _sample(self, **kwargs):
        raise NotImplementedError

    def _update_posterior(self, arm_index, observed_reward):
        raise NotImplementedError

    @property
    def estimated_arm_means(self):
        raise NotImplementedError

    @property
    def estimated_arm_sds(self):
        raise NotImplementedError


class ThompsonBernoulli(ThompsonSampling):
    def __init__(self, bandit, alpha_prior, beta_prior):
        ThompsonSampling.__init__(self, bandit, alpha=alpha_prior, beta=beta_prior)

    def _sample(self, alpha, beta):
        return scipy.stats.beta.rvs(alpha, beta, size=1)

    def _update_posterior(self, arm_index, observed_reward):
        if observed_reward == 1:
            self.posterior_params[arm_index]['alpha'] += 1
        else:
            self.posterior_params[arm_index]['beta'] += 1

    @property
    def estimated_arm_means(self):
        mean = lambda params: \
                params['alpha'] / (params['alpha'] + params['beta'])
        return np.array([ mean(params) for params in self.posterior_params])
    @property
    def estimated_arm_sds(self):
        sd = lambda params: np.sqrt(
                    (params['alpha'] / (params['alpha'] + params['beta'])) *
                    (1 - params['alpha'] / (params['alpha'] + params['beta']))
                )
        return np.array([sd(params) for params in self.posterior_params])

class ThompsonGaussianKnownSigma(ThompsonSampling):
    def __init__(self, bandit, sigma, mu_prior, sigma_prior,
            memory_multiplier=0.9):
        ThompsonSampling.__init__(self, bandit, mu=mu_prior,
                sigma2=sigma_prior ** 2)
        self.sigma2 = sigma ** 2
        self.sufficient_statistics = [{'n': 0, 'xsum': 0} for _ in range(self.num_arms)]
        self.memory_multiplier = memory_multiplier

    def _sample(self, mu, sigma2):
        return np.random.normal(loc=mu, scale=np.sqrt(sigma2))

    def _update_posterior(self, arm_index, observed_reward):
        sigma2 = self.sigma2

        for index in range(self.num_arms):
            old_n = self.sufficient_statistics[index]['n']
            old_xsum = self.sufficient_statistics[index]['xsum']
            mu_prior = self.prior_params[index]['mu']
            sigma2_prior = self.prior_params[index]['sigma2']

            if index == arm_index:
                new_n = self.memory_multiplier * old_n + 1
                new_xsum = self.memory_multiplier * old_xsum + observed_reward
            else:
                new_n = self.memory_multiplier * old_n
                new_xsum = self.memory_multiplier * old_xsum

            new_sigma2_posterior = 1 / (1 / sigma2_prior + new_n / sigma2)
            new_mu_posterior = (mu_prior / sigma2_prior + new_xsum / sigma2) * new_sigma2_posterior

            self.sufficient_statistics[index]['n'] = new_n
            self.sufficient_statistics[index]['xsum'] = new_xsum
            self.posterior_params[index]['mu'] = new_mu_posterior
            self.posterior_params[index]['sigma2'] = new_sigma2_posterior

    @property
    def estimated_arm_means(self):
        return np.array([params['mu'] for params in self.posterior_params])

    @property
    def estimated_arm_sds(self):
        return np.array([self.sigma2 + params['sigma2']
            for params in self.posterior_params])




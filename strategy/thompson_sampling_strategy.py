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
        estimated_arm_means (ndarray): posterior mean
        estimated_arm_sds (ndarray): posterior standard deviation
    """
    def __init__(self, bandit, **kwargs):
        self.bandit = bandit
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
            observed_rewards[i] = self._pull_arm(index_arms_pulled[i])
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

    def _pull_arm(self, arm_index):
        return self.bandit.pull_arm(arm_index)

    def _sample(self, **kwargs):
        raise NotImplementedError

    def _update_posterior(self, arm_index, observed_reward):
        raise NotImplementedError

    @property
    def num_arms(self):
        return bandit.num_arms

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
                    params['alpha'] * params['beta'] /
                    ( (params['alpha'] + params['beta'])**2 *
                      (params['alpha'] + params['beta'] + 1) )
                )
        return np.array([sd(params) for params in self.posterior_params])

class ThompsonGaussianKnownSigma(ThompsonSampling):
    def __init__(self, bandit, sigma, mu_prior, sigma_prior,
            memory_multiplier=1.0):
        ThompsonSampling.__init__(self, bandit, mu=mu_prior,
                sigma2=sigma_prior ** 2)
        self.sigma2 = sigma ** 2
        self.sufficient_statistics = [{'n': 0, 'xsum': 0} for _ in range(self.num_arms)]
        if memory_multiplier <= 0 or memory_multiplier > 1:
            raise ValueError("Memory Multiplier should be in (0,1]")
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
        return

    @property
    def estimated_arm_means(self):
        return np.array([params['mu'] for params in self.posterior_params])

    @property
    def estimated_arm_sds(self):
        return np.array([params['sigma2'] for params in self.posterior_params])

class ThompsonGaussian(ThompsonSampling):
    def __init__(self, bandit, mu_prior, nu_prior, alpha_prior, beta_prior,
            memory_multiplier=1.0):
        ThompsonSampling.__init__(self, bandit,
                mu=mu_prior, nu=nu_prior, alpha=alpha_prior, beta=beta_prior)
        self.sufficient_statistics = {
                'n': np.zeros(self.num_arms),
                'xsum': np.zeros(self.num_arms),
                'x2sum': np.zeros(self.num_arms),
                }
        if memory_multiplier <= 0 or memory_multiplier > 1:
            raise ValueError("Memory Multiplier should be in (0,1]")
        self.memory_multiplier = memory_multiplier

    def _sample(self, mu, nu, alpha, beta):
        sigma2 = scipy.stats.invgamma.rvs(a=alpha, scale=beta)
        return np.random.normal(loc=mu, scale=np.sqrt(sigma2/nu))

    def _update_posterior(self, arm_index, observed_reward):
        # Update Sufficient Statistics of arm_index
        self.sufficient_statistics['n'][arm_index] += 1
        self.sufficient_statistics['xsum'][arm_index] += observed_reward
        self.sufficient_statistics['x2sum'][arm_index] += observed_reward**2

        # Memory Multiplier Decay
        self.sufficient_statistics['n'] *= self.memory_multiplier
        self.sufficient_statistics['xsum'] *= self.memory_multiplier
        self.sufficient_statistics['x2sum'] *= self.memory_multiplier

        # Update posterior params -> Complicated due to posterior_params
        #                            (really should use pandas)
        for index in range(self.num_arms):
            n = self.sufficient_statistics['n'][index]
            xsum = self.sufficient_statistics['xsum'][index]
            x2sum = self.sufficient_statistics['x2sum'][index]
            mu = self.prior_params[index]['mu']
            nu = self.prior_params[index]['nu']
            alpha = self.prior_params[index]['alpha']
            beta = self.prior_params[index]['beta']

            # Terminate early if no data to update
            if n == 0:
                continue

            # Conjugate Prior Update
            self.posterior_params[index]['mu'] = (mu * nu + xsum)/(nu + n)
            self.posterior_params[index]['nu'] = nu + n
            self.posterior_params[index]['alpha'] = alpha + n/2.0
            self.posterior_params[index]['beta'] = \
                    beta + 0.5*(x2sum - xsum**2/n) + \
                    0.5*(nu*n)/(nu+n)*(xsum/n - mu)**2
        return

    @property
    def estimated_arm_means(self):
        return np.array([params['mu'] for params in self.posterior_params])

    @property
    def estimated_arm_sds(self):
        return np.array([
            np.sqrt(params['beta'] / (params['alpha']-1) / params['nu'])
            for params in self.posterior_params])




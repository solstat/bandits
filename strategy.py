import numpy as np

class Strategy:
    """ Base Strategy Class

    Args:

    Attributes:


    """
    def __init__(self, bandit, **kwargs):
        raise NotImplementedError

    def fit(self, iterations, **kwargs):
        """ Fit

        Args:
            iterations (int): number of lsjdflajdf
            **kwargs: other

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        raise NotImplementedError

class EpsilonGreedy(Strategy):
    """ Epislon Greedy Strategy
    
    """
    def __init__(self, bandit, epsilon, **kwargs):
        self.bandit = bandit
        self.epsilon = epsilon
        self.arms_list = bandit.get_arms_list()
#        self.estimate_arm_reward = np.zeros(self.arms_list.size)
        return

    def fit(self, iterations, **kwargs):
        """ Fit

        Args:
            iterations (int): number of iterations
            **kwargs: other

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        assert(iterations >= len(self.arms_list))
        iteration = 0
        
        rewards_per_arm = [[] for i in range(len(self.arms_list))]
        rewards = [None] * iterations
        arms_pulled = [None] * iterations
                      
        estimated_arm_means = np.zeros((len(self.arms_list), iterations))
        

        def pull_arm_index(arm_index, iteration):
            arm = self.arms_list[arm_index]
            reward = self.bandit.pull_arm(arm)
            
            arms_pulled[iteration] = arm
            rewards[iteration] = reward
            rewards_per_arm[arm_index].append(reward)
            
            estimated_arm_means[:,iteration] = np.array(
                    [np.mean(a) for a in rewards_per_arm]
                    )
            return
            
        # Pull each arm once
        scan_order = np.arange(len(self.arms_list))
        np.random.shuffle(scan_order)
        for arm_index in scan_order:
            pull_arm_index(arm_index, iteration)
            iteration += 1
        
        # Epsilon Greedy
        while(iteration < iterations):
            if(np.random.rand() < self.epsilon):
                # Explore
                arm_index = np.random.randint(0, len(self.arms_list))
                pull_arm_index(arm_index, iteration)
                iteration += 1
                
            else:
                # Greedy
                arm_index = np.argmax([np.mean(a) for a in rewards_per_arm])
                pull_arm_index(arm_index, iteration)
                iteration += 1
                
            
        out_dict = dict(
                rewards = rewards,
                arms_pulled = arms_pulled,
                estimated_arm_means = estimated_arm_means,
                )
        return out_dict

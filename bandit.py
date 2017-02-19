class Bandit:
    def __init__(self):
        raise NotImplementedError

    def get_arms_list(self):
        """ The arms in the bandit

                Returns:
                    a list of the arms in the bandit. These
                    should be hashable and have a str representation.
        """
        raise NotImplementedError

    def pull_arm(self, arm):
        """ Pulls an arm of the ba

                Returns:
                    a double corresponding to the reward of pulling the
                    arm of the bandit.
        """
        raise NotImplementedError

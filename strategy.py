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

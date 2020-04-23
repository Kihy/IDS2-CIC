import unittest
from vae import kullback_leibler_loss
import numpy as np
import scipy


def kl_divergence(mu1,mu2, std1,std2):
    """
    calculates kl divergence between two distributions numerically

    Args:
        mu1(float): mean of distribution 1
        mu2(float): mean of distribution 2
        std1(float): standard deviation of distribution 1
        std2(float): standard deviation of distribution 2

    Returns:
        float: the divergence

    """

    return np.log(std2/std1)+(std1**2+(mu1-mu2)**2)/(2*std2**2)-0.5


class Test(unittest.TestCase):

    def test_kl_divergence(self):
        """
        tests kl divergence calculation with different formulas

        Returns:
            None

        """

        mean=[0., 1., 1., 1.]
        std=[1., 0.2, 0.5, 0.3]


        for i in range(len(mean)):


            numerical_kl=kl_divergence(mean[i], 0, std[i],1)
            analytical_kl=kullback_leibler_loss(np.array([mean[i]]),np.array([np.log(std[i]**2)])).numpy()
            print(numerical_kl)
            self.assertAlmostEqual(numerical_kl, analytical_kl, places=4)
if __name__ == '__main__':
    unittest.main()

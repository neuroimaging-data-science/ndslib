import numpy as np


def gaussian_kernel(x=20, sigma=4):
    """
    Construct a 2D Gaussian kernel for image processing

    Parameters
    ----------
    x : int, optional
        The number of pixels on a side for the filter.
        Default : 20
    sigma : float, optional
        The standard deviation parameter for the Gaussian.
        Default : 4

    Returns
    -------
    gauss : ndarray
        Contains the values of the 2D Gaussian normalized
        to sum to 1.
    """

    im = np.meshgrid(range(-x//2, x//2), range(-x//2, x//2))
    gauss = np.exp(-(im[0] ** 2 + im[1] ** 2)/(2 * sigma ** 2))
    gauss = gauss / np.sum(gauss)
    return gauss

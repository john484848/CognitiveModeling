import numpy as np

def multivariate_normal_density(x, mean, covariance):
    """
    Compute the density of a multivariate normal distribution.
    
    Parameters:
        x: numpy array, N-dimensional vector for which density is to be calculated
        mean: numpy array, N-dimensional vector representing the mean of the distribution
        covariance: numpy array, N by N covariance matrix of the distribution
    
    Returns:
        density: float, the density of the multivariate normal distribution at point x
    """
    # Number of dimensions
    N = len(mean)
    
    # Compute determinant and inverse of the covariance matrix
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)
    
    # Compute exponent term
    exponent = -0.5 * np.dot(np.dot((x - mean).T, inv_covariance), (x - mean))
    
    # Compute constant term
    constant = 1 / ((2 * np.pi) ** (N/2) * det_covariance ** 0.5)
    
    # Compute density
    density = constant * np.exp(exponent)
    
    return density

# Example usage
mean = np.array([0, 0])
covariance = np.array([[1, 0.5], [0.5, 2]])
x = np.array([1, 1])

density = multivariate_normal_density(x, mean, covariance)
print("Density:", density)

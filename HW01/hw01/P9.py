import numpy as np
from scipy.stats import multivariate_normal

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

# Define scenarios
scenarios = [
    {"mean": np.array([0, 0]), "covariance": np.eye(2)},
    {"mean": np.array([0, 0]), "covariance": np.diag([1, 2])},
    {"mean": np.array([0, 0]), "covariance": np.array([[2, 0.5], [0.5, 1]])}
]

# Test each scenario
for i, scenario in enumerate(scenarios, 1):
    mean = scenario["mean"]
    covariance = scenario["covariance"]
    print(f"Scenario {i}:")
    print("Mean:", mean)
    print("Covariance:")
    print(covariance)
    
    # Generate a random point
    x = np.random.rand(len(mean)) * 10  # Random point within 10 in each dimension
    
    # Compute density using custom function
    custom_density = multivariate_normal_density(x, mean, covariance)
    
    # Compute density using scipy's function
    scipy_density = multivariate_normal.pdf(x, mean, covariance)
    
    print("Custom Density:", custom_density)
    print("SciPy Density:", scipy_density)
    print("\n")

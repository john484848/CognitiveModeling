import numpy as np
import matplotlib.pyplot as plt

def simulate_diffusion(v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    v     : float
        The drift rate (rate of information uptake)
    a     : float
        The boundary separation (decision threshold).
    beta  : float in [0, 1]
        Relative starting point (prior option preferences)
    tau   : float
        Non-decision time (additive constant)
    dt    : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time: float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a 
        binary decision (c - int) 
    """

    # Inits (process starts at relative starting point)
    y = beta * a
    num_steps = tau
    const = scale*np.sqrt(dt)

    # Loop through process and check boundary conditions
    while (y <= a and y >= 0) and num_steps <= max_time:

        # Perform diffusion equation
        z = np.random.randn()
        y += v*dt + const*z

        # Increment step counter
        num_steps += dt

    if y >= a:
        c = 1
    else:
        c = 0
    return (round(num_steps, 3), c)


def simulate_diffusion_n(num_sims, v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """
    Simulates n realizations of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    num_sims : int
        The number of simulations/observations we want to run
    v        : float
        The drift rate (rate of information uptake)
    a        : float
        The boundary separation (decision threshold).
    beta     : float in [0, 1]
        Relative starting point (prior option preferences)
    tau      : float
        Non-decision time (additive constant)
    dt       : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale    : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time : float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    [(x1, c1),(x2,c2),...,(xn,cn)] - a list of tuples of response time (y - float) and a 
        binary decision (c - int) 
    """
    data = np.zeros((num_sims, 2))
    for n in range(num_sims):
        data[n, :] = simulate_diffusion(v, a, beta, tau, dt, scale, max_time)
    return data

def parameter_experiment(parameter):
    """
    Simulates 2000 realizations of the diffusion process given
    a set of parameters and a step size `dt` for 25 different values
    of the parameter. We assume that v = 1, a = 2, beta = 0.5, and 
    tau = 0.7. We then vary a given parameter. 

    Parameters:
    -----------
    parameter: string
        The parameter of the drift diffusion model we want to vary

    Returns:
    --------
    No returns but plots the means and standard deviation of the 2000 observations for each of the 25 varied parameters.
    """

    # We set up the initial values for the drift diffusion model
    xlabel = 'Starting Points'
    offset = 0
    v = 1
    a = 3
    beta = 0.5
    tau = 0.7

    # Given the value of the parameter, we will modify the offset variable, which will be used to 
    # create the range of values and xlabel that will be used for plotting purposes.
    if parameter == 'v':
        offset = 50
        xlabel = 'Drift Rates'
    elif parameter == 'a':
        offset = 250
        xlabel = 'Boundary Separations'
    elif parameter == 'tau':
        offset = 40
        xlabel = 'Non-Decision Times'

    # We create the range of values for the parameters
    parameter_range = (np.arange(25)*4 + offset)/100

    # We will hold the 2000 simulaions of each of the 25 different values of the parameters, which returns
    # the response time and the decision made
    complete_data = np.zeros((25, 2000, 2))

    # We will create lists to hold the means of each parameter value for the lower and upper boundary
    lower_boundary_means = []
    upper_boundary_means = []

    # We will create lists to hold the standard deviations of each parameter value for the lower and upper boundary
    lower_boundary_SDs = []
    upper_boundary_SDs = []
    index = 0

    # We will loop through each of the parameter values
    for parameter_value in parameter_range:    
        # Depending on the given parameter we want to vary, we will change the value of the parameter
        if parameter == 'v':
            v = parameter_value
        elif parameter == 'a':
            a = parameter_value
        elif parameter == 'beta':
            beta = parameter_value
        else:
            tau = parameter_value

        # We will then set up the diffusion simulation with the given parameters
        params = {
            'v': v,
            'a': a,
            'beta': beta,
            'tau': tau
        }
        complete_data[index, :] = simulate_diffusion_n(2000, **params)

        # We will find the indices that correspond to the lower boundary and the upper boundary
        lower_boundary_indices = np.where(complete_data[index,:, 1] == 0)
        upper_boundary_indices = np.where(complete_data[index,:, 1] == 1)
            
        # We then grab the response times that correspond to the lower boundary and the upper boundary
        lower_boundaries = complete_data[index, lower_boundary_indices, 0]
        upper_boundaries = complete_data[index, upper_boundary_indices, 0]

        # We will then find the means of the response times for the lower and upper boundary
        lower_boundary_mean = np.mean(lower_boundaries)
        upper_boundary_mean = np.mean(upper_boundaries)

        # We will add the means to the list of total means
        lower_boundary_means.append(lower_boundary_mean)
        upper_boundary_means.append(upper_boundary_mean)

        # We will then find the standard deviations of the response times for the lower and upper boundary
        lower_boundary_SD = np.std(lower_boundaries)
        upper_boundary_SD = np.std(upper_boundaries)

         # We will add the standard deviations to the list of total means
        lower_boundary_SDs.append(lower_boundary_SD)
        upper_boundary_SDs.append(upper_boundary_SD)

        index+=1

    # We will set up a 1 by 2 image setup
    f, axarr = plt.subplots(1, 2, figsize=(10,5))

    # For the left image, we will plot the mean distribution for both the lower and upper boundaries
    axarr[0].plot(parameter_range, lower_boundary_means, '-', color="blue")
    axarr[0].plot(parameter_range, upper_boundary_means, '-', color="red")
    axarr[0].legend(['Lower Bondary', 'Upper Boundary'], fontsize=16)
    axarr[0].set_xlabel(xlabel, fontsize=16)
    axarr[0].set_ylabel('Boundary Means', fontsize=16)

    # For the right image, we will plot the standard deviation distribution for both the lower and upper boundaries
    axarr[1].plot(parameter_range, lower_boundary_SDs, '-', color="blue")
    axarr[1].plot(parameter_range, upper_boundary_SDs, '-', color="red")
    axarr[1].legend(['Lower Bondary', 'Upper Boundary'], fontsize=16)
    axarr[1].set_xlabel(xlabel, fontsize=16)
    axarr[1].set_ylabel('Boundary Standard Deviations', fontsize=16)
    plt.show()

# We will vary the drift rate(v) for 25 different values each with 2000 observations
parameter_experiment('v')

# We will vary the boundary separation(a) for 25 different values each with 2000 observations
parameter_experiment('a')

# We will vary the starting point(beta) for 25 different values each with 2000 observations
parameter_experiment('beta')

# We will vary the non-decision times(tau) for 25 different values each with 2000 observations
parameter_experiment('tau')
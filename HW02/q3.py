import numpy as np
from numba import jit
@jit
def simulate_ffi(v, x0, a,tau,alpha,sigma=1, dt=1e-3,max_time=10.):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    v     : np.ndarray
        The drift rates (rates of information uptake)
    x0    : np.ndarray
        The starting points
    a     : float
        The boundary separation (decision threshold).
    beta  : float in [0, inf+)
        Inhibition parameter
    kappa : float in [0, 1]
        Leakage parameter
    tau   : float [0 , inf+)
        Non-decision time (additive constant)
    dt    : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    max_time: float, optional (default: 10.)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a 
        binary decision (c - int) 
    """

    # Inits (process starts at relative starting point)
    num_steps = tau
    xlog=[]
    x = x0.copy()
    xlog.append((0,x))
    const = sigma*np.sqrt(dt)
    assert x.shape[0] == v.shape[0]
    J = x0.shape[0]
    Jconst=len(v)

    # Loop through process and check boundary conditions
    while num_steps <= max_time:
        
        # Sample random noise variate
        z = np.random.randn(J)
        # Loop through accumulators
        for j in range(J):
            # LCA equation
            dx_j = (v[j]-((alpha/(Jconst-1))*sum(v)))*dt + const*z[j]
            x[j] = max(x[j] + dx_j, 0)

        # Increment step counter
        num_steps += dt
        xlog.append((num_steps,x))

        # Check for boundary hitting
        if any(x >= a):
            break
    
    return (round(num_steps, 3), x.argmax())
parameters = {
    'v': np.array([1., 1., 3.]),
    'x0': np.zeros(3),
    'a': 1.,
    'tau': 0.5,
    'alpha': 0.6,
}
parameters2 = {
    'v': np.array([1., 1., 3.]),
    'x0': np.zeros(3),
    'a': 1.,
    'tau': 0.5,
    'beta': 0.6,
    'kappa': 0.2
}
print(simulate_ffi(**parameters))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from numba import jit
#@jit
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
    (x, c, log) - a tuple of response time (y - float) and a 
        binary decision (c - int) and log of accumulators at each time step
    """

    # Inits (process starts at relative starting point)
    num_steps = tau
    xlog={}
    x = x0.copy()
    xlog[0]=x.copy()
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
        xlog[num_steps]=x.copy()

        # Check for boundary hitting
        if any(x >= a):
            break
    
    return (round(num_steps, 3), x.argmax(),xlog)
def graph():
    """
    generates graph of the accumulators for run
    """
    parameters = {
            'v': np.array([1., 1., 3.,2.]),
            'x0': np.zeros(4),
            'a': 1.,
            'tau': 0.5,
            'alpha': 0.6,
            }
    res=simulate_ffi(**parameters)
    intrestedValue=res[2]
    intrestedValueTransform={}
    for i in range(parameters["x0"].size):
        intrestedValueTransform["Accumulator "+ str(i)]=[]
    for x in intrestedValue:
        accumcount=0
        for i in intrestedValue[x]:
            intrestedValueTransform["Accumulator "+ str(accumcount)].append(i)
            accumcount+=1
    data=pd.DataFrame.from_dict(intrestedValueTransform)
    sns.set_theme()
    sns.relplot(data=data, kind="line")
    plt.show()


parameters = {
        'v': np.array([1., 1., 3.,2.]),
        'x0': np.zeros(4),
        'a': 1.,
        'tau': 0.5,
        'alpha': 0.6,
    }
log=np.array([])
for i in range(40):
    parameters['alpha']+=2
    res=simulate_ffi(**parameters)
    log=np.append(log, res[0])
print(np.mean(log))
parameters = {
    'v': np.array([1., 1., 3.,2.]),
    'x0': np.zeros(4),
    'a': 1.,
    'tau': 0.5,
    'alpha': 0.6,
}
log=np.array([])
for i in range(40):
    res=simulate_ffi(**parameters)
    log=np.append(log, res[0])
print(np.mean(log))

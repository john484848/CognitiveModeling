import numpy as np
import math
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def normal_normal(mu_0,sigma_0,x,sigma):
    prior=stats.norm(mu_0,sigma_0**2)
    postirior=stats.norm((mu_0/(sigma_0**2))*((mu_0/(sigma_0**2)))+(sum(x)/(sigma**2)),(1/(sigma**2))+(len(x)/(sigma_0**2)))
    prior_density = prior.pdf(x)
    posterior_density =  postirior.pdf(x)
    return {"Prior":prior_density,"Posterior":posterior_density,"x":x}

num_samples = 1000
mean = 50.0
sigma_0 = 10.0
samples = np.random.normal(loc=0.0, scale=sigma_0, size=num_samples)
sigma = np.std(samples)
b=normal_normal(mean, sigma_0, samples, sigma)
b=pd.DataFrame.from_dict(b)
sns.histplot(data=b,x="x",y="Prior")
plt.savefig("Priorq5.png")
plt.clf()
sns.histplot(data=b,x="x",y="Posterior")
plt.savefig("Posteriorq5.png")

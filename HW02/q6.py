import numpy as np 
import stan
import pandas as pd
stancode="""
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}parameters {
  real alpha;
  real beta;
  real sigma;
}
model {
    sigma ~ inv_gamma(1,1);
    alpha ~ normal(0,10);
    beta ~ normal(0,10);
    y ~ normal(alpha +beta*x,sigma); 
}
"""
N = 100
alpha = 2.3
sigma = 2. 
slope = 4. 
x = np.random.normal(size=N)
y = alpha + slope * x + sigma * np.random.normal(size=N)
data={"x":x,"y":y,"N":N}
p=stan.build(stancode,data=data)
fit=p.sample()
print(y)

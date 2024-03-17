import numpy as np 
import stan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
stancode="""
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}parameters {
  real alpha;
  real beta;
  real <lower=0> sigma;
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
df=pd.DataFrame.from_dict(data)
p=stan.build(stancode,data=data)
fit=p.sample(num_chains=4, num_samples=N)
results=fit.to_frame()
for col in results.columns:
    print(col)
dic={}
s=[]
for i in results["alpha"]:
    s.append(abs(i-alpha))
dic["Alpha diff"]=s
s=[]
p=[]
for l in results["beta"]:
    p.append(abs(l-slope))
for i in results["sigma"]:
    s.append(abs(i-sigma))
dic["Sigma diff"]=s
dic["Beta diff"]=p
gr=pd.DataFrame.from_dict(dic)
sns.set_theme()
sns.relplot(data=gr, kind="line")
plt.savefig('fig.png')

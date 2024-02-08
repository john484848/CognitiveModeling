import numpy as np
import sys
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
TransformLambda= lambda x: 2*x -1
TransformVecorize=np.vectorize(TransformLambda)
def sim(n):
    x= TransformVecorize(np.random.random(size=n))
    y= TransformVecorize(np.random.random(size=n))
    pointcord=list(range(n))
    WithinLambda= lambda index: x[index]**2 + y[index]**2<=1
    withinL=list(filter(WithinLambda, pointcord))
    piapprox=4*(len(withinL)/n)
    return piapprox
def error_cal(x):
    return abs(x-np.pi)
n=int(sys.argv[1])
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(sim)(i) for i in range(1,n))
resultsfinal = Parallel(n_jobs=num_cores)(delayed(error_cal)(i) for i in results)
dataframer= pd.DataFrame(data={"Number of Points": list(range(1,n)), "Error":resultsfinal})
sns.set_theme()
sns.relplot(data=dataframer, x="Number of Points", y="Error", kind="line")
plt.show()
#for i in range(1,n):
#    piapproxl.append(sim(i))
#print(piapproxl)

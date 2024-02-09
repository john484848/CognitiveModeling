import numpy as np
from matplotlib import pyplot as plt
import math

# probability of a true positive(test correctly identifies the individual having the disease)
sensitivity = 0.95

# probability of a true negative(test correctly identifies the individual not having the disease)
specificity = 0.90

# probabilty of having the disease
prior = 0.01

# posterior: probability of having the disease given a positive test
# p(x = having the disease | y = positive test) = p(x = having the disease, y = positive test)/p(y = positive test)
# p(x = having the disease, y = positive test)/p(y = positive test) = (prior)(sensitivity)/(((prior)(sensitivity)) + ((1-prior)(1-specificity))))
# (prior)(sensitivity)/(((prior)(sensitivity)) + ((1-prior)(1-specificity)))) = (0.01)(0.95)/(0.95)(0.01)+(0.1)(0.99)

# First Case: The posteriror probability of actually having the disease given a positive test as a function of prior
# Assume fixed sensitivity and specificity

X = np.linspace(0, 1, 1000)
Y = (X*sensitivity)/((sensitivity*X)+((1-specificity)*(1-X)))

plt.plot(X, Y)
plt.xlabel("Prior Probability")
plt.ylabel("Posterior Probability")
plt.title("Posterior Probability as a function of Prior Probability")
plt.show()

# Second Case: The posteriror probability of actually having the disease given a positive test as a function of sensitivity
# Assume fixed prior and specificity

Y = (prior*X)/((X*prior)+((1-specificity)*(1-prior)))

plt.plot(X, Y)
plt.xlabel("Sensitivity Probability")
plt.ylabel("Posterior Probability")
plt.title("Posterior Probability as a function of Sensitivity Probability")
plt.show()

# Third Case: The posteriror probability of actually having the disease given a positive test as a function of specificity
# Assume fixed prior and sensitivity

Y = (prior*sensitivity)/((sensitivity*prior)+((1-X)*(1-prior)))

plt.plot(X, Y)
plt.xlabel("Specificity Probability")
plt.ylabel("Posterior Probability")
plt.title("Posterior Probability as a function of Specificity Probability")
plt.show()

# Bonus Problems
  
# First Case: The posteriror probability of actually having the disease given a positive test as a function of prior and sensitivity

ax = plt.axes(projection ='3d')

X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z = (X*Y)/((Y*X)+((1-specificity)*(1-X)))
 
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Prior Probability')
ax.set_ylabel('Sensitivity Probability')
ax.set_zlabel('Posterior Probability')
ax.set_title('Posterior Probability as a function of Prior and Sensitivity Probability')
plt.show()

# Second Case: The posteriror probability of actually having the disease given a positive test as a function of prior and specificity

ax = plt.axes(projection ='3d')

X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z = (X*sensitivity)/((sensitivity*X)+((1-Y)*(1-X)))
 
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Prior Probability')
ax.set_ylabel('Specificity Probability')
ax.set_zlabel('Posterior Probability')
ax.set_title('Posterior Probability as a function of Prior and Specificity Probability')
plt.show()

# Third Case: The posteriror probability of actually having the disease given a positive test as a function of sensitivity and specificity

ax = plt.axes(projection ='3d')
 
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z = (prior*X)/((X*prior)+((1-Y)*(1-prior)))
 
ax.plot_surface(X, Y, Z)
ax.set_xlabel('Sensitivity Probability')
ax.set_ylabel('Specificity Probability')
ax.set_zlabel('Posterior Probability')
ax.set_title('Posterior Probability as a function of Sensitivity and Specificity Probability')
plt.show()
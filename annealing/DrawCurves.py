import numpy as np

import matplotlib.pyplot as plt

# Define the x range
x = np.linspace(0, 1, 500)

# Define the functions
tlin = 1 - x
texp = np.exp(-5 * x)
tcos = np.cos(np.pi/2 * x**2)
texp2 = np.exp(-10 * x)
tcos2 = np.cos(np.pi/2 * x**4)
tplateau = 0.5 - (1.6*x - 0.8)**3
tsigmoid = 1 - 1 / (1 + np.exp(5 - 10 * x))

# Create the plot
plt.figure()

# Plot each function
plt.plot(x, tlin, label='lin')
plt.plot(x, texp, label='exp')
plt.plot(x, tcos, label='cos')
plt.plot(x, texp2, label='exp2')
plt.plot(x, tcos2, label='cos2')
plt.plot(x, tplateau, label='plateau')
plt.plot(x, tsigmoid, label='sigmoid')

# Add a legend
plt.legend()

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Annealing Curves')

# Show the plot
plt.show()

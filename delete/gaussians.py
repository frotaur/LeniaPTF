import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, mean, std,beta=1):
    return (beta / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# Define the x values
x = np.linspace(0, 1, 400)

# Parameters for the three Gaussians
means = [0.2]
stds = [0.04]
betas = [1]
# Sum of the three Gaussians
y = 2*gaussian(x, means[0], stds[0])-1 
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Growth function', color='green')
plt.xlabel('r')
plt.ylabel('')
plt.xticks([0,0.2, 0.5, 0.8, 1])
plt.yticks([])
plt.grid(False)  # Disables the grid
plt.legend()
plt.show()
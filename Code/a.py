import numpy as np

# Define the range of numbers
numbers = np.arange(1, 9)

# Generate 10,000 samples with uniform distribution
samples = np.random.choice(numbers, size=10000, replace=True)

# Output the first 10 samples to verify
print(samples[:10])
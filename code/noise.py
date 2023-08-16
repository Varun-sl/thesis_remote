import pandas as pd
import numpy as np

# Load the original CSV file into a Pandas DataFrame
original_df = pd.read_csv('leaf_features.csv')

# Define parameters for white noise
noise_mean = 0
noise_std = 0.01  # Adjust the standard deviation based on your needs

# Generate white noise with the same shape as the dataset
noise = np.random.normal(noise_mean, noise_std, size=original_df.shape)

# Add white noise to the dataset and ensure positivity
noisy_data = original_df + np.abs(noise)

# Round the second column to whole numbers
noisy_data.iloc[:, 1] = noisy_data.iloc[:, 1].round().astype(int)

# Save the noisy dataset to a new CSV file
noisy_df = pd.DataFrame(noisy_data, columns=original_df.columns)
noisy_df.to_csv('leaf_features_noise.csv', index=False)

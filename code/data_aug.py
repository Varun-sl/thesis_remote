import pandas as pd
import numpy as np

# Load your CSV data
data = pd.read_csv(r'../dataset/leaf_features_latest_backup.csv')


# Define data augmentation transformations
def augment_data(row, num_augmentations=9):
    augmented_rows = []

    for _ in range(num_augmentations):
        augmented_row = row.copy()

        # Add Gaussian noise to leaf count
        augmented_row['Leaf Count'] += np.random.randint(1, 4)
        augmented_rows.append(augmented_row.copy())

        # Example 2: Multiply leaf area by a random factor
        random_factor = np.random.uniform(0.9, 1.1)
        augmented_row['Leaf Area Covered'] *= random_factor
        augmented_rows.append(augmented_row.copy())

        # Example 3: Multiply weight by a random factor
        random_factor = np.random.uniform(0.9, 1.1)
        augmented_row['Weight'] *= random_factor

        # Ensure weight is not equal to 0
        if augmented_row['Weight'] <= 0:
            augmented_row['Weight'] = 0.1

        augmented_rows.append(augmented_row.copy())


    return augmented_rows


# Apply data augmentation
augmented_data = []
for index, row in data.iterrows():
    augmented_rows = augment_data(row)
    augmented_data.extend(augmented_rows)

# Convert augmented data back to DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Ensure constraints: Make sure leaf count is a positive integer, leaf area and weight are positive floats
augmented_df['Leaf Count'] = augmented_df['Leaf Count'].astype(int).apply(lambda x: max(x, 0))
augmented_df['Leaf Area Covered'] = augmented_df['Leaf Area Covered'].apply(lambda x: max(x, 0))
augmented_df['Weight'] = augmented_df['Weight'].apply(lambda x: max(x, 0))

# Save the augmented data to a new CSV file
augmented_df.to_csv(r'leaf_features_aug.csv', index=False)
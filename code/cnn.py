import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class CustomCNN(nn.Module):
    def __init__(self, in_features, hidden_features=128):
        super(CustomCNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(r'leaf_features_noise.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        leaf_count = self.data.iloc[idx, 1]  # Replace with actual column index for leaf count
        leaf_area = self.data.iloc[idx, 2]   # Replace with actual column index for leaf area
        return torch.tensor([leaf_count, leaf_area], dtype=torch.float32)

# Load your trained CustomCNN model
custom_model = CustomCNN(in_features=2, hidden_features=128)
custom_model.load_state_dict(torch.load(r"CNN.pth"))
custom_model.eval()

# Load custom dataset
transform = None
batch_size = 50
test_dataset = CustomDataset(csv_file=r'leaf_features_noise_5.csv', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        for input_sample in inputs:
            output = custom_model(input_sample)
            predictions.append(output)

#predictions = torch.cat(predictions)
print("Predicted Biomass:")
y_test = []
count = 0
pred1=[]
for pred in predictions:
    count = count +1
    print(pred.item())
    pred1.append(pred.item())
print(count)

import csv

# Open the CSV file for reading
with open(r'leaf_features_noise.csv', 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)

    # Initialize a list to store values from the first column
    weight = []

    # Iterate through each row in the CSV and append the value from the first column to the list
    for row in csvreader:
        if row:  # Check if the row is not empty
            weight.append(row[0])  # Append the value from the first column



# Print the extracted values
for value in weight:
    print(value)
# print(len(weight[1:]))
# print(len(predictions))
plt.scatter(weight[1:], pred1)
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Biomass (grams)')
plt.ylabel('Predicted Biomass (grams)')
plt.title('Biomass Estimation - Multi-Layer Perceptron')
#plt.ylim(0.05, 0.15)
plt.show()
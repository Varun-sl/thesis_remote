import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Import metrics
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("leaf_features_aug.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[["Leaf Count", "Leaf Area Covered"]], data["Weight"], test_size=0.2, random_state=42
)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # Convert y_train to a numpy array
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the MLP model using nn.Module
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = MLP()

# Define loss and optimizer
criterion = nn.MSELoss()  # Using Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 7000
batch_size = 250
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor.view(-1, 1))
    r2 = r2_score(y_test, test_outputs)  # Calculate R-squared score
    mse = mean_squared_error(y_test, test_outputs)  # Calculate MSE
    mae = mean_absolute_error(y_test, test_outputs)  # Calculate MAE

print(f"Test Loss: {test_loss.item():.4f}")
print(f"R-squared: {r2:.4f}")  # Print R2 score
print(f"Mean Squared Error (MSE): {mse:.4f}")  # Print MSE
print(f"Mean Absolute Error (MAE): {mae:.4f}")  # Print MAE

# Make predictions on the test set
with torch.no_grad():
    test_predictions = model(X_test_tensor)

# Rest of the code remains unchanged

# Plot the actual biomass values and predicted biomass values
plt.scatter(y_test, test_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Biomass (grams)')
plt.ylabel('Predicted Biomass (grams)')
plt.title('Biomass Estimation - Multi-Layer Perceptron')
# plt.ylim(0.05, 0.15)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Define your custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # Input: leaf count and leaf area
        self.fc2 = nn.Linear(128, 1)  # Output: predicted biomass

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        leaf_count = self.data.iloc[idx, 1]  # Replace with actual column index for leaf count
        leaf_area = self.data.iloc[idx, 2]   # Replace with actual column index for leaf area
        label = self.data.iloc[idx, 0]       # Biomass column index
        return torch.tensor([leaf_count, leaf_area], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Hyperparameters
batch_size = 100
learning_rate = 0.001
num_epochs = 5000

# Instantiate the model
model = CustomCNN()

# Choose loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load custom dataset
transform = None  # No image data, so no transform needed
dataset = CustomDataset(csv_file=r'leaf_features_noise.csv', transform=transform)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss}")

# Save trained model
torch.save(model.state_dict(), r'CNN.pth')
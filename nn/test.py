
## Test Module ##
## Takes only the test portion of the dataset ## 
## Runs full set of test data and prints accuracy ##
## Also lets you print output for a single song at a time ## 

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

# Load test data
data_path = "/volumes/data/final_data.pt"  # Ensure this matches your training data file
dataset = torch.load(data_path)

# Extract test chromagrams (X) and tag labels (Y)
X = torch.tensor(dataset['chromagrams'], dtype=torch.float32).unsqueeze(1)  # Add channel dim
Y = torch.tensor(dataset['tags'], dtype=torch.float32)

# Use only the test set (last 20%)
split = int(0.8 * len(X))
test_X, test_Y = X[split:], Y[split:]

# Create DataLoader
batch_size = 32
test_loader = data.DataLoader(list(zip(test_X, test_Y)), batch_size=batch_size, shuffle=False)

# Define CNN Model (must match the trained model)
class ChromagramCNN(nn.Module):
    def __init__(self):
        super(ChromagramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3 * 625, 512)
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))  # Multi-label classification

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChromagramCNN().to(device)
model.load_state_dict(torch.load("chromagram_cnn_model.pt"))
model.eval()

# Evaluate Model
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary labels
        correct += (predicted == labels).sum().item()
        total += labels.numel()

print(f"Test Accuracy: {correct / total:.4f}")

# Predict for a single sample
def predict_single(track_idx):
    """Predicts the mood tags for a single track in the test dataset."""
    input_tensor = test_X[track_idx].unsqueeze(0).to(device)  # Add batch dim
    with torch.no_grad():
        output = model(input_tensor)
    predictions = (output > 0.5).int().cpu().numpy().flatten()
    print(f"Predicted Tags: {predictions}")

# Run a single prediction (change index as needed)
predict_single(0)

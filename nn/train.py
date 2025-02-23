<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 9978d68 (Add preliminary draft of NN folder)
## Train module ##
## Splits dataset into train and test with an 80/20 split ##
## Allows you to change number of epochs and prints test metrics ##

<<<<<<< HEAD
=======
>>>>>>> 7e3ba94 (Add initial drafts of nn folder)
=======
>>>>>>> 9978d68 (Add preliminary draft of NN folder)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# Load your preprocessed data
data_path = "/volumes/data/final_data.pt"  # Update with your actual file path
dataset = torch.load(data_path)  # Assumes dict with 'chromagrams' and 'tags'

# Extract chromagrams (X) and tag labels (Y)
X = torch.tensor(dataset['chromagrams'], dtype=torch.float32).unsqueeze(1)  # Add channel dim
Y = torch.tensor(dataset['tags'], dtype=torch.float32)  # Binary labels

# Train-test split (80-20 split)
split = int(0.8 * len(X))
train_X, test_X = X[:split], X[split:]
train_Y, test_Y = Y[:split], Y[split:]

# Create PyTorch DataLoader
class ChromagramDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

batch_size = 32  # Adjust based on memory
train_loader = data.DataLoader(ChromagramDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(ChromagramDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

# CNN Model (from previous response)
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

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChromagramCNN().to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "chromagram_cnn_model.pt")
print("Model training complete and saved.")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        correct += (predicted == labels).sum().item()
        total += labels.numel()

print(f"Test Accuracy: {correct / total:.4f}")

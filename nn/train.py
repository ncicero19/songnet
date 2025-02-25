## Train module ##
## Splits dataset into train and test with an 80/20 split ##
## Allows you to change number of epochs and prints test metrics ##

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from cnn import ChromagramCNN

# Load your preprocessed data
data_path = "/volumes/data/final_data.pt"  
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

batch_size = 16  # Adjust based on memory
train_loader = data.DataLoader(ChromagramDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(ChromagramDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChromagramCNN().to(device)

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)                                                     

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
        epsilon = 1e-6
        outputs = torch.clamp(outputs, min=epsilon, max=1-epsilon)
        loss = criterion(outputs, labels.squeeze(2))
        labels = labels.squeeze(2)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.clamp(outputs, min=epsilon, max=1-epsilon)
            labels = labels.squeeze(2)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.numel()
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "cnn_model.pt")
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

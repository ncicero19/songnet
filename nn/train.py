## Train module ##
## Splits dataset into train and test with an 80/20 split ##
## Allows you to change number of epochs and prints test metrics ##

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from cnn import ChromagramCNN
import matplotlib.pyplot as plt

# Load your preprocessed data
data_path = "/volumes/data/final_data.pt"  
dataset = torch.load(data_path)  # Assumes dict with 'chromagrams' and 'tags'

nan_rows = dataset['chromagrams'].isnan().any(axis=1).any(axis=1)
chromagrams = dataset['chromagrams']
tags = dataset['tags']
# Filter out the rows with NaN values from both chromagrams and tags
filtered_chromagrams = chromagrams[~nan_rows, :, :]
filtered_tags = tags[~nan_rows, :, :]

# Extract chromagrams (X) and tag labels (Y)
X = filtered_chromagrams.clone().detach().float().unsqueeze(1)  # Add channel dim 
Y = filtered_tags.clone().detach().float()  # Binary labels

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

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChromagramCNN().to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)                                                     
print("Starting training...")

# Initialize lists to store epoch, loss, and accuracy values
epochs = []
accuracies_1 = []
accuracies_2 = []
accuracies_3 = []

# Training loop
num_epochs = 50  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # For accuracy metrics 
    batch_count = 1
    total_correct = 0
    total_elements = 0
    perfect_matches = 0
    adjusted_accuracy_total = 0
    track_count = 0
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero previous gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        # Clamp outputs to avoid division by 0
        epsilon = 1e-6
        outputs = torch.clamp(outputs, min=epsilon, max=1-epsilon)
        labels = labels.squeeze(2)
        # Print various output and label metrics for debugging
        print("----------------------------------------------------------------------------")
        print(f"Epoch {epoch+1}/{num_epochs}, Batch: {batch_count}")
        if torch.isnan(outputs).any().item():
            print("Any NaN outputs?", torch.isnan(outputs).any().item())
        if torch.isnan(labels).any().item():
            print("Any NaN labels?", torch.isnan(labels).any().item())
        if torch.isinf(outputs).any().item():
            print("Any Inf outputs?", torch.isinf(outputs).any().item())
        if torch.isinf(labels).any().item():
            print("Any Inf labels?", torch.isinf(labels).any().item()) 

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Loss: {loss.item():.4f}")
        # Implement accuracy metrics 
        predicted = (outputs > 0.5).float()
        batch_track_count = labels.size(0)
            
        # Standard accuracy calculation
        batch_correct = (predicted == labels).sum().item()
        total_correct += batch_correct
        batch_elements = labels.numel()
        total_elements += batch_elements
        print(f"Accuracy 1: {round((batch_correct / batch_elements)*100, 1)}%")    
        
        # Adjusted accuracy calculation
        correct_outputs = (predicted * labels).sum(dim=1)
        total_outputs = predicted.sum(dim=1)
        total_correct_tags = labels.sum(dim=1)

        denominator = total_outputs + total_correct_tags - correct_outputs
        valid_cases = denominator > 0
        adjusted_accuracy = torch.where(valid_cases, correct_outputs / denominator, torch.zeros_like(denominator))
        batch_adjusted_accuracy = adjusted_accuracy.sum().item()
        adjusted_accuracy_total += batch_adjusted_accuracy
        print(f"Accuracy 2: {round((batch_adjusted_accuracy / batch_track_count)*100, 1)}%")

         # Perfect accuracy calculation (tracks where all tags are correct)
        batch_perfect_matches = (predicted == labels).all(dim=1).sum().item()
        perfect_matches += batch_perfect_matches
        print(f"Accuracy 3: {round((batch_perfect_matches / batch_track_count)*100, 1)}%")
        
        batch_count += 1
        track_count += batch_track_count
        
        
    # Per Epoch
    avg_loss = running_loss / len(train_loader)
    standard_accuracy = total_correct / total_elements
    perfect_accuracy = perfect_matches / track_count
    adjusted_accuracy_final = adjusted_accuracy_total / track_count
    print("-------------------------------------------------------------------------------")
    print(f"EPOCH {epoch+1}/{num_epochs} COMPLETE")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Everything Accuracy: {(standard_accuracy*100):.4f}%, Adjusted Accuracy: {(adjusted_accuracy_final*100):.4f}%, Perfect Accuracy: {(perfect_accuracy*100):.4f}")
    # Append metrics 
    epochs.append(epoch+1)
    accuracies_1.append(standard_accuracy*100)
    accuracies_2.append(adjusted_accuracy_final*100)
    accuracies_3.append(perfect_accuracy*100)

# Save model
torch.save(model.state_dict(), "cnn_model.pt")
print("Model training complete and saved.")

plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracies_1, label="Accuracy 1", marker="o")
plt.plot(epochs, accuracies_2, label="Accuracy 2", marker="s")
plt.plot(epochs, accuracies_3, label="Accuracy 3", marker="^")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Metrics vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()
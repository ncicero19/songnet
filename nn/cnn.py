<<<<<<< HEAD
## CNN Architecture to be used by the train and test modules ## 


=======
>>>>>>> 7e3ba94 (Add initial drafts of nn folder)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChromagramCNN(nn.Module):
    def __init__(self):
        super(ChromagramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 625, 512)  # Adjusted for downsampled feature size
        self.fc2 = nn.Linear(512, 26)  # Output layer (26 mood tags)

    def forward(self, x):
        # Conv layers with activation, batch norm, pooling, and dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for multi-label classification

        return x

# Model initialization
model = ChromagramCNN()
print(model)


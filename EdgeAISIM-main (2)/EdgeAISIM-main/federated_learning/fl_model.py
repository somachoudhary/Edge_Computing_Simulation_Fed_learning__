import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(Net, self).__init__()
        
        # Fully connected layers with non-linearity
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Output layer (logits)
        return F.log_softmax(x, dim=1)  # Log-Softmax for classification

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # Define layers
        self.l1 = nn.Linear(input_size, hidden_size)  # Input -> Hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Hidden -> Hidden
        self.l3 = nn.Linear(hidden_size, num_classes)  # Hidden -> Output
        
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the layers
        out = self.l1(x)  # First layer
        out = self.relu(out)  # Apply ReLU
        out = self.l2(out)  # Second layer
        out = self.relu(out)  # Apply ReLU again
        out = self.l3(out)  # Output layer (no activation function here)
        
        return out

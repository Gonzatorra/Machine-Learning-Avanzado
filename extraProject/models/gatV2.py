import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm

class HabitabilityGATV2(torch.nn.Module):
    def __init__(self, hidden_channels, heads=4):
        super(HabitabilityGATV2, self).__init__() 
        torch.manual_seed(111)

        # Layer 1: Capture complex relationships using GATv2
        # GATv2Conv provides dynamic attention for better feature weighing
        self.conv1 = GATv2Conv(8, hidden_channels, heads=heads, concat=True)
        self.bn1 = BatchNorm(hidden_channels * heads) # Batch Normalization for training stability
        
        # Layer 2: Aggregate information from all attention heads
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.bn2 = BatchNorm(hidden_channels)

        # Extra linear layer to provide more depth before final classification
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        # 1. First layer processing with Batch Normalization and Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        # Increased dropout to 0.4 to improve regularization and prevent overfitting
        x = F.dropout(x, p=0.4, training=self.training) 

        # 2. Second layer processing
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        
        # 3. Final classification output
        return self.classifier(x)
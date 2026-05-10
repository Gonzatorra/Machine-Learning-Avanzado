import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class HabitabilityGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(HabitabilityGNN, self).__init__()
        # Set seed for reproducibility
        torch.manual_seed(111)
        
        # Layer 1: Input (8 features) -> Hidden
        self.conv1 = SAGEConv(8, hidden_channels)
        # Layer 2: Hidden -> Hidden
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Final classification head (2 output classes: 0 or 1)
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        # 1. First Message Passing step
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 2. Second Message Passing step
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        # 3. Final prediction per node (planet)
        out = self.classifier(x)
        
        return out


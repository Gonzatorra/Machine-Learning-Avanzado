import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HabitabilityGAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads=4):
        super(HabitabilityGAT, self).__init__()
        torch.manual_seed(111)

        # Layer 1: Multi-head attention layer
        # heads act like different eyes looking at various features simultaneously
        self.conv1 = GATConv(8, hidden_channels, heads=heads, concat=True)
        
        # Layer 2: Second attention layer
        # Input is (hidden_channels * heads) because the previous layer concatenates head outputs
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)

        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        # 1. First pass with attention and Dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)

        # 2. Second pass (heads=1, so concatenation is not needed)
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 3. Final Classification
        out = self.classifier(x)
        return out
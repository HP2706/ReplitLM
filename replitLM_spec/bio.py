import torch 
import numpy as np

class BioLinear(nn.Module):
    def __init__(self, in_dim, out_dim, in_fold=1, out_fold=1, in_head=1, out_head=1, device = 'cpu'):
        super(BioLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias = True, device = device)
        self.in_fold = in_fold
        self.out_fold = out_fold
        self.in_head = in_head
        self.out_head = out_head
        assert in_dim % in_fold == 0
        assert out_dim % out_fold == 0
        #compute in_cor, shape: (in_dim)
        in_dim_fold = int(in_dim/in_fold)
        out_dim_fold = int(out_dim/out_fold)
        self.in_coordinates = torch.tensor(list(np.linspace(1/(2*in_dim_fold), 1-1/(2*in_dim_fold), num=in_dim_fold))*in_fold, dtype=torch.float)
        self.out_coordinates = torch.tensor(list(np.linspace(1/(2*out_dim_fold), 1-1/(2*out_dim_fold), num=out_dim_fold))*out_fold, dtype=torch.float)
        
    def forward(self, x):
        return self.linear(x)

    def from_linear(layer: nn.Linear, device) -> 'BioLinear':
        """Creates a BioLinear layer from a linear layer."""
        return BioLinear(layer.in_features, layer.out_features, in_fold=1, out_fold=1, device= device)


# GNN with bias
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCN2Conv

class simpleGCN2(nn.Module):
    def __init__(self,numlayer,inchannel,midchannel,outchannel,isshared_weights=True,
                 isNormalize=True,isResidual=True):
        super().__init__()

        self.GNNlayers = nn.ModuleList()
        self.residual = isResidual
        
        self.inLayer = nn.Linear(inchannel,midchannel)
        self.inActivate = nn.ReLU()
        
        for i in range(numlayer-1):
            self.GNNlayers.append(GCN2Conv(midchannel,midchannel,
                                           shared_weights=isshared_weights,normalize=isNormalize))
            self.GNNlayers.append(nn.LayerNorm(midchannel))
            self.GNNlayers.append(nn.ReLU())

        self.outLayer = nn.Linear(midchannel,outchannel)
        
    def forward(self,x,edge,edgeweight=None):
        x = x.squeeze(0)
        x = self.inActivate(self.inLayer(x))
        x2 = x
        x_0 = x
        for i,l in enumerate(self.GNNlayers):
            if i % 3 != 0 :

                if self.residual and i % 6 == 5:
                    x = x + x2

                x = l(x)

                if self.residual and i % 6 == 5:
                    x2 = x

            else :
                x = l(x,x_0,edge_index=edge,edge_weight=edgeweight)

        output = self.outLayer(x)
        
        output = output.unsqueeze(0)
        x = x.unsqueeze(0)
        return x,output
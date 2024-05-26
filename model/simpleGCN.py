# GNN with bias
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

class simpleGCN(nn.Module):
    def __init__(self,numlayer,inchannel,midchannel,outchannel,isBias=True,isNormalize=True,isResidual=True):
        super().__init__()

        self.GNNlayers = nn.ModuleList()
        self.residual = isResidual
        
        self.inLayer = nn.Linear(inchannel,midchannel)
        self.inActivate = nn.ReLU()
        
        for i in range(numlayer-1):
            self.GNNlayers.append(GCNConv(midchannel,midchannel,
                                           bias=isBias,normalize=isNormalize))
            self.GNNlayers.append(nn.ReLU())

        self.outLayer = nn.Linear(midchannel,outchannel)
        
    def forward(self,x,edge,edgeweight=None):
        x = self.inActivate(self.inLayer(x))
        x2 = x
        for i,l in enumerate(self.GNNlayers):
            if i & 1 :

                if self.residual and i % 4 == 3:
                    x = x + x2

                x = l(x)

                if self.residual and i % 4 == 3:
                    x2 = x

            else :
                x = l(x,edge_index=edge,edge_weight=edgeweight)

        output = self.outLayer(x)
        return x,output
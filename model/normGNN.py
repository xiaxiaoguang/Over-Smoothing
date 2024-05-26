# GNN with bias
import torch
import torch.nn as nn

from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.norm import PairNorm

class normGNN(nn.Module):
    def __init__(self,numlayer,inchannel,midchannel,outchannel,isBias=True,isResidual=True):
        super().__init__()

        self.GNNlayers = nn.ModuleList()
        self.residual = isResidual
        
        self.inLayer = nn.Linear(inchannel,midchannel)
        self.inActivate = nn.ReLU()
        
        for i in range(numlayer-1):
            self.GNNlayers.append(GraphConv(midchannel,midchannel,
                                           bias=isBias))
            self.GNNlayers.append(nn.LayerNorm(midchannel))            
            self.GNNlayers.append(nn.ReLU())

        self.outLayer = nn.Linear(midchannel,outchannel)
        
    def forward(self,x,edge,edgeweight=None):
        x = self.inActivate(self.inLayer(x))
        x2 = x
        for i,l in enumerate(self.GNNlayers):
            if i % 3 != 0 :

                if self.residual and i % 6 == 5:
                    x = x + x2

                x = l(x)

                if self.residual and i % 6 == 5:
                    x2 = x

            else :
                x = l(x,edge_index=edge,edge_weight=edgeweight)

        output = self.outLayer(x)
        return x,output
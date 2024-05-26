# GNN with bias
import torch
import torch.nn as nn

from torch_geometric.nn.conv import GCNConv

class nakedGCN(nn.Module):
    def __init__(self,numlayer,inchannel,midchannel,outchannel,isBias=True,isNormalize=True,isResidual=True):
        super().__init__()

        self.GNNlayers = nn.ModuleList()
        self.residual = isResidual
        
        self.inLayer = GCNConv(inchannel,midchannel,bias=isBias,normalize=isNormalize)
        self.inActivate = nn.ReLU()
        
        for i in range(numlayer-2):
            self.GNNlayers.append(GCNConv(midchannel,midchannel,
                                           bias=isBias,normalize=isNormalize))
            self.GNNlayers.append(nn.ReLU())

        self.outLayer = GCNConv(midchannel,outchannel,bias=isBias,normalize=isNormalize)
        
    def forward(self,x,edge,edgeweight=None):
        x = self.inActivate(self.inLayer(x,edge))
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

        output = self.outLayer(x,edge)
        return x,output
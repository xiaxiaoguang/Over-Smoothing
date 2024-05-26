from torch_geometric.datasets import Planetoid,Amazon,BitcoinOTC
from torch.utils.data import Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self,root,name,type,isTrain,maskRate):
        super().__init__()
        
        self.isTrain = isTrain
        self.maskRate= maskRate

        if type=="Amazon":
            self.dataset = Amazon(root,name)
        elif type == "Planetoid":
            self.dataset = Planetoid(root,name)
        elif type=="BitcoinOTC" :
            self.dataset = BitcoinOTC(root)
        else :
            raise TypeError
        
    def __len__(self):
        return self.dataset.len()
    
    def __getitem__(self, index):
        g_dataset = self.dataset.get(index)
        
        if self.isTrain:
            mask = torch.arange(start=self.maskRate*g_dataset.num_nodes,end=g_dataset.num_nodes,dtype=torch.int)
        else :
            mask = torch.arange(end=self.maskRate*g_dataset.num_nodes,dtype=torch.int)

        return g_dataset.x,g_dataset.y,g_dataset.edge_index,mask

# Photos Amazon
# Cora Planetoid
# PubMed Planetoid
    
if __name__ == "__main__":
    a = GraphDataset("data","","BitcoinOTC")
    print(a.__len__())
    x,y,e = a.__getitem__(1)
    print(x.shape,y.shape,e.shape)
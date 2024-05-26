import torch

def Eng2(node_feature,edge_index):
    num_nodes = node_feature.shape[0]
    ret = 0
    for i,u in enumerate(node_feature):
        source = (edge_index[0] == i)
        target = (edge_index[1] == i)
        neighbors = torch.cat((edge_index[1][source], edge_index[0][target]))
        dif = node_feature[neighbors]-u
        l2_norms = torch.sum(torch.norm(dif, dim=-1, p=2))
        ret += l2_norms
    ret /= num_nodes
    return ret

def Eng(node_feature,edge_index):
    num_nodes = node_feature.shape[0]
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    source_diffs = node_feature[source_nodes] - node_feature[target_nodes]
    target_diffs = node_feature[target_nodes] - node_feature[source_nodes]
    diffs = torch.cat([source_diffs, target_diffs], dim=0)
    l2_norms = torch.norm(diffs, dim=-1, p=2)
    total_l2_norms = torch.sum(l2_norms)
    ret = total_l2_norms / num_nodes # 应该加根号的，但是无所谓了
    return ret

def accuracy(x,y):
    c = (x.argmax(dim=-1) == y).to(torch.int8)
    ret = c.sum()/c.shape[0]
    return ret
# if __name__=="__main__":
#     # a = GraphDataset("data","photo","Amazon")
#     x,y,e = a.__getitem__(1)
#     print(Eng(x,e),Eng2(x,e))

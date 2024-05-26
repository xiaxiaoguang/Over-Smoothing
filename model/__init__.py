from .simpleGCN import simpleGCN as simpleGCN
from .simpleGCN2 import simpleGCN2 as simpleGCN2

from .normGCN import normGCN as normGCN
from .simpleGNN import simpleGNN as simpleGNN
from .normGNN import normGNN as normGNN
from .nakedGCN import nakedGCN as nakedGCN
from .simpleGAT import simpleGAT as simpleGAT

__all__ = ["simpleGCN","normGCN","normGNN",
           "simpleGNN","nakedGCN","simpleGAT","simpleGCN2"]

import easytorch
from argparse import ArgumentParser

if __name__ =="__main__":
    parser = ArgumentParser(description="Graph Nueral Network HW7")
    parser.add_argument("-c", "--cfg", required="simpleGNN.py", help="training config")
    parser.add_argument("-g","--gpus",default="1",help="training gpus")
    args = parser.parse_args()
    easytorch.launch_training(cfg=args.cfg,gpus=args.gpus,node_rank=0)

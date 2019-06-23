import torch
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, graph_reader, feature_reader, target_reader

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    clustering_machine = ClusteringMachine(args, graph, features, target)
    clustering_machine.decompose()
    gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
    gcn_trainer.train()
    gcn_trainer.test()

if __name__ == "__main__":
    main()

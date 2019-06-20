import torch
from layers import StackedGCN
import random
from tqdm import tqdm

class ClusterGCNTrainer(object):
     
    def __init__(self, args, features, clustering_machine, target):
        self.args = args
        self.features = features
        self.clustering_machine = clustering_machine
        self.target = target
        self.create_model()
        self.transfer_data()


    def create_model(self):
        self.model = StackedGCN(self.args, self.features.shape[1], self.target.shape[1])
    
    def transfer_data(self):
        self.features = torch.FloatTensor(self.features)
        self.target = torch.LongTensor(self.target)

    def train(self):
        self.model.eval()
        for epoch in tqdm(range(self.args.epochs)):
            random.shuffle(self.clustering_machine.clusters)
            for cluster in self.clustering_machine.clusters:
                edges = self.clustering_machine.sg_edges[cluster]
                macro_nodes = self.clustering_machine.sg_nodes[cluster]
                train_nodes = self.clustering_machine.sg_train_nodes[cluster]
                test_nodes = self.clustering_machine.sg_test_nodes[cluster]
                features = self.features[macro_nodes,:]
                print(features.shape)
                target = self.target[macro_nodes,:]
                fp = self.model(edges,features)

import metis
import torch
import random
import community
import networkx as nx
from sklearn.model_selection import train_test_split

class ClusteringMachine(object):
     
    def __init__(self, args, graph):
        self.args = args
        self.graph = graph

    def decompose(self):
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        elif self.args.clustering_method == "louvain":
            print("\nLouvain graph clustering started.\n")
            self.louvain_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_cluster_membership_mapping()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        (_, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def louvain_clustering(self):
        self.cluster_membership = community.best_partition(self.graph)
        self.clusters = list(set(self.cluster_membership.values()))

    def general_cluster_membership_mapping(self):
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        
        for cluster in self.clusters:
            subgraph = nx.subgraph(self.graph, [node for node in self.graph.nodes() if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in subgraph.nodes()]
            mapper = {node: i for i, node in enumerate(self.sg_nodes[cluster])}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()),test_size = self.args.test_ratio)


    def transfer_edges_and_nodes(self):
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster] ).view(2,-1)
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            

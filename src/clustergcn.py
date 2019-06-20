import torch
import random
from layers import StackedGCN
from torch.autograd import Variable
from tqdm import trange

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
        self.target = torch.FloatTensor(self.target)

    def calculate_loss(self, predictions, target):
        loss_matrix = -(target*torch.log(predictions)+(1-target)*torch.log(1-predictions))
        average_loss = torch.mean(loss_matrix)
        return average_loss

    def do_weight_update(self, cluster):
        edges = self.clustering_machine.sg_edges[cluster]
        macro_nodes = self.clustering_machine.sg_nodes[cluster]
        train_nodes = self.clustering_machine.sg_train_nodes[cluster]
        features = self.features[macro_nodes,:]
        target = self.target[macro_nodes,:]
        predictions = self.model(edges, features)
        average_loss = self.calculate_loss(predictions[train_nodes,:], target[train_nodes,:])
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        edges = self.clustering_machine.sg_edges[cluster]
        macro_nodes = self.clustering_machine.sg_nodes[cluster]
        test_nodes = self.clustering_machine.sg_test_nodes[cluster]
        features = self.features[macro_nodes,:]
        target = self.target[macro_nodes,:]
        target = target[test_nodes,:]
        predictions = self.model(edges, features)
        predictions = predictions[test_nodes,:]
        threshold = Variable(torch.Tensor([self.args.threshold]))
        predictions = (predictions > threshold).float() * 1
        accuracy_scores = (predictions==target).type(torch.FloatTensor)
        accuracy = torch.mean(accuracy_scores).item()
        node_count = test_nodes.shape[0]
        return node_count, accuracy

    def train(self):
        print("Training started.\n")
        epochs = trange(self.args.epochs, desc = "Train Loss:")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_weight_update(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
            epochs.set_description("Train Loss: %g" % round(average_loss,4))

    def test(self):
        self.model.eval()
        node_count_seen = 0
        accumulated_hits = 0
        for cluster in self.clustering_machine.clusters:
            node_count, accuracy = self.do_prediction(cluster)
            node_count_seen = node_count_seen + node_count
            accumulated_hits = accumulated_hits +accuracy*node_count
        overall_accuracy = accumulated_hits/node_count_seen
        print("\nTest Accuracy: %g.\n" % round(overall_accuracy,4))
            
            





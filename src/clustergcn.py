import torch
import random
import numpy as np
from layers import StackedGCN
from tqdm import trange, tqdm
from torch.autograd import Variable
from sklearn.metrics import f1_score
class ClusterGCNTrainer(object):
     
    def __init__(self, args, features, clustering_machine, target):
        self.args = args
        self.features = features
        self.clustering_machine = clustering_machine
        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.transfer_data()


    def create_model(self):
        self.model = StackedGCN(self.args, self.features.shape[1], self.target.shape[1])
        self.model = self.model.to(self.device)
    
    def transfer_data(self):
        self.features = torch.FloatTensor(self.features)
        self.target = torch.FloatTensor(self.target)

    def calculate_loss(self, predictions, target):
        loss_matrix = -(target*torch.log(predictions)+(1-target)*torch.log(1-predictions))
        average_loss = torch.mean(loss_matrix)
        return average_loss

    def do_weight_update(self, cluster):
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.features[macro_nodes,:].to(self.device)
        target = self.target[macro_nodes,:].to(self.device)
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
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.features[macro_nodes,:].to(self.device)
        target = self.target[macro_nodes,:].to(self.device)
        target = target[test_nodes,:]
        prediction = self.model(edges, features)
        prediction = prediction[test_nodes,:]
        threshold = Variable(torch.Tensor([self.args.threshold])).to(self.device)
        prediction = (prediction > threshold).float() * 1
        return prediction, target

    def train(self):
        print("Training started.\n")
        epochs = trange(self.args.epochs, desc = "Train Loss")
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
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions)

    def score(self):
        mask = self.targets.mean(axis=1)!=0
        f_1_score = f1_score(self.targets[mask,:], self.predictions[mask,:], average='micro') 
        print("\nTest F1-Score: %g.\n" % round(f_1_score,4))

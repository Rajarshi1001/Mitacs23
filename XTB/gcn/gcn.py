import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import pandas as pd
import shutil
import os
import re
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

PATH = os.getcwd()
DATA_DIR = os.path.join(os.path.abspath(os.path.join(PATH, os.pardir)), "XTBData")
EPOCHS = 5000
BATCH_SIZE = 128
LEARNING_RATE = 0.009801507765569738 
WIDTH = 1024
FUNNEL_SIZE = 8
WEIGHT_DECAY = 0.0008
DROPOUT_RATE = 0.04
AMSGRAD = False
KERNEL_SIZE = 3
NUM_NODES = 2048
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

train_mae, val_mae = [], []
train_mse, val_mse = [], []
train_graphs, val_graphs, test_graphs = [], [], []

def preprocess():

    train_path = os.path.join(DATA_DIR, sys.argv[1])
    val_path = os.path.join(DATA_DIR, sys.argv[2])
    # test_path = os.path.join(DATA_DIR, sys.argv[3])
    
    # reading the train file
    train_data = pd.read_csv(train_path, header = 0)
    val_data = pd.read_csv(val_path, header = 0)
    # test_data = pd.read_csv(test_path, header = 0)
    feat_cols = train_data.columns[2:]

    for row_idx, train_row in train_data.iterrows():

        train_id = train_row["id"]
        train_features = train_row[feat_cols]
        train_targets = train_row["bandgap"]
        if train_targets > 1.0:
            graph = nx.Graph()
            for idx in range(NUM_NODES):
                graph.add_node(idx, feature = train_features[idx], target = train_targets)
            distances = np.zeros((NUM_NODES, NUM_NODES))
            for i in range(NUM_NODES):
                intersection = 0
                union = 0
                for j in range(i+1, NUM_NODES):
                    if train_features[i] == 1 and train_features[j] == 1:
                        intersection += 1
                        union += 1
                    elif train_features[i] == 1 or train_features[j] == 1:
                        union += 1
                    elif train_features[i] == 0 or train_features[j] == 1:
                        union += 1
                    distance = intersection / union
                    distances[i, j] = distance
                    distances[j, i] = distance
            for i in range(NUM_NODES):
                nearest_neighbours = np.argsort(distances[i])[:5]
                for j in nearest_neighbours:
                    graph.add_edge(i,j)
            train_graphs.append(graph)

    for row_idx, val_row in val_data.iterrows():

        val_id = val_row["id"]
        val_features = val_row[feat_cols]
        val_targets = val_row["bandgap"]
        if val_targets > 1.0:
            graph = nx.Graph()
            for idx in range(NUM_NODES):
                graph.add_node(idx, feature = val_features[idx], target = val_targets)
            distances = np.zeros((NUM_NODES, NUM_NODES))
            for i in range(NUM_NODES):
                intersection = 0
                union = 0
                for j in range(i+1, NUM_NODES):
                    if val_features[i] == 1 and val_features[j] == 1:
                        intersection += 1
                        union += 1
                    elif val_features[i] == 1 or val_features[j] == 1:
                        union += 1
                    elif val_features[i] == 0 or val_features[j] == 0:
                        union += 1
                    distance = intersection / union
                    distances[i, j] = distance
                    distances[j, i] = distance
            for i in range(NUM_NODES):
                nearest_neighbours = np.argsort(distances[i])[:5]
                for j in nearest_neighbours:
                    graph.add_edge(i,j)
            val_graphs.append(graph)
    # with open(train_path, "r") as file:
    #     for line in file:
    #         buffer = line.strip().split(",")
    #         val_features = buffer[2:]
    #         val_id = buffer[0]
    #         val_targets = buffer[1]
    #         if val_target > 1.0:
    #             val_dataset.append([np.float32(val_features), np.float32([val_targets]), str(val_id)])

    
    # for row_idx, test_row in test_data.iterrows():

    #     test_id = test_row["id"]
    #     test_features = test_row[feat_cols]
    #     test_targets = test_row["bandgap"]
    #     if test_targets > 1.0:
    #         test_dataset.append([np.float32(test_features), np.float32([test_targets]), str(test_id)])


    return train_graphs, val_graphs, test_graphs

def main(argv):

    

    # converting networkx graphs into pytorch geomtric objects
    train_datalist, val_datalist = [], []
    for train_graph in train_graphs:
        edge_index = torch.tensor(list(train_graph.edges)).t().contiguous()
        x = torch.tensor([train_graph.nodes[node]["feature"] for node in train_graph.nodes]).view(-1, NUM_NODES).float()
        target = torch.tensor([train_graph.nodes[node]["target"] for node in train_graph.nodes]).view(-1, OUTPUT_SIZE).float()
        data = Data(x=x, edge_index=edge_index, y=target)
        train_datalist.append(data)

    for val_graph in val_graphs:
        edge_index = torch.tensor(list(val_graph.edges)).t().contiguous()
        x = torch.tensor([val_graph.nodes[node]["feature"] for node in val_graph.nodes]).view(-1, NUM_NODES).float()
        target = torch.tensor([val_graph.nodes[node]["target"] for node in val_graph.nodes]).view(-1, OUTPUT_SIZE).float()
        data = Data(x=x, edge_index=edge_index, y=target)
        val_datalist.append(data)
    
    trainloader = DataLoader(train_datalist, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_datalist, batch_size=BATCH_SIZE, shuffle=True)
    model = GCNNet(NUM_NODES, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    loss_fn = nn.MSELoss()

    Tloss = []
    Vloss = []

    for epoch in range(EPOCHS):
        print("Epoch : {}".format(epoch))
        print("--------------------------")
        tloss = train(model, trainloader, loss_fn, optimizer)
        Tloss.append(tloss)
        print("Epoch {} training loss : {}".format(epoch, tloss))
        if tloss == min(Tloss):
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, "model_min_T.pth")
        vloss = validate(model, valloader, loss_fn)
        Vloss.append(vloss)
        print("Epoch {} validation loss : {}".format(epoch, vloss))
        if vloss == min(Vloss):
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, "model_min_V.pth")

    weights_path = "gcnnet.pth"
    torch.save(model.state_dict(), weights_path)
    model.eval()

    with open("gcntrainout.txt", "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(trainloader):
            x = in_fea.to(device)
            pred, target = model(x)
            for i in zip(ids, target[:, 0].detach().tolist(), pred[:, 0].detach().tolist()):
                outfile.write(",".join(map(str, i)) + "\n")

    with open("gcnvalout.txt", "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(valloader):
            x = in_fea.to(device)
            pred, target = model(x)
            for i in zip(ids, target[:, 0].detach().tolist(), pred[:, 0].detach().tolist()):
                outfile.write(",".join(map(str, i)) + "\n")

    with open("losses.txt", "w") as outfile:
        for i in zip(Tloss, Vloss):
            outfile.write(",".join(map(str, i)) + "\n")

    val_target, val_pred = [], []
    train_target, train_pred = [], []

    with open("gcnvalout.txt", "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            val_target.append(float(buffer[1]))
            val_pred.append(float(buffer[2]))

    with open("gcntrainout.txt", "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            train_target.append(float(buffer[1]))
            train_pred.append(float(buffer[2]))

    train_mae.append(mean_absolute_error(np.array(train_pred), np.array(train_target)))
    val_mae.append(mean_absolute_error(np.array(val_pred), np.array(val_target)))
    train_mse.append(mean_squared_error(np.array(train_pred), np.array(train_target)))
    val_mse.append(mean_squared_error(np.array(val_pred), np.array(val_target)))

    plt.plot(Tloss, label="Train")
    plt.plot(Vloss, label="Val")
    plt.legend(loc="upper right")
    plt.savefig('losses.png')

    plt.figure(figsize=(20, 10))
    plt.title("target vs prediction on test")
    plt.ylabel("Target values")
    plt.xlabel("Predicted values")
    plt.scatter(train_pred, train_target, color="blue", label="data points", s=40)
    plt.scatter(val_pred, val_target, color="red", label="data points", s=40)
    plt.legend(["Train", "Validation"])
    plt.plot([min(val_target), max(val_target)], [min(val_target), max(val_target)], color="red", label="Reference line")
    plt.grid()
    plt.savefig("prediction.png")
    print("datapoints: {}".format(len(val_pred + train_pred)))

def crossval(n):
    print("Training across {} sets".format(n))
    for _ in range(n):
        main(sys.argv)
    print("---------- MAE ----------")
    print("Train mae: {}".format(train_mae))
    print("Validation mae: {}".format(val_mae))
    print("--------- MSE----------")
    print("Train mse: {}".format(train_mse))
    print("Validation mse: {}".format(val_mse))


class GCNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, data):
        x, edge_index, target = data.x, data.edge_index, data.y
        x = self.gelu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x, target

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_graphs = 0
    for data in dataloader:
        optimizer.zero_grad()
        output, target = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_graphs += 1
    return total_loss / num_graphs

def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    num_graphs = 0
    with torch.no_grad():
        for data in dataloader:
            output, target = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            num_graphs += 1
    return total_loss / num_graphs


if __name__ == "__main__":
    train_graphs, val_graphs, test_graphs = preprocess()
    # train_dataset = train_dataset + val_dataset
    print("The train dataset consists of {} molecules".format(len(train_graphs)))
    print("The validation dataset consists of {} molecules".format(len(val_graphs)))

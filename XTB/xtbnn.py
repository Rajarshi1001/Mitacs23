import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import plotly.express as px

# DATA_DIR = os.path.join("XTB","XTBData")
DATA_DIR = "XTB/XTBData"

EPOCHS = 5000
BATCH_SIZE = 128
LEARNING_RATE = 0.009801507765569738 
WIDTH = 1024
FUNNEL_SIZE = 8
WEIGHT_DECAY = 0.0008
DROPOUT_RATE = 0.2
AMSGRAD = False
KERNEL_SIZE = 3
INPUT_SIZE = 2048
OUTPUT_CHANNELS = 16
INPUT_CHANNELS = 1

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

train_mae, val_mae = [], []
train_rmse, val_rmse = [], []
train_dataset, val_dataset, test_dataset = [], [], []

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
            train_dataset.append([np.float32(train_features), np.float32([train_targets]), str(train_id)])

    for row_idx, val_row in val_data.iterrows():
        val_id = val_row["id"]
        val_features = val_row[feat_cols]
        val_targets = val_row["bandgap"]
        if val_targets > 1.0:
            val_dataset.append([np.float32(val_features), np.float32([val_targets]), str(val_id)])
    
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


    return train_dataset, val_dataset, test_dataset

def main(argv): 

    train_dataset, val_dataset, test_dataset = preprocess()
    # train_dataset = train_dataset + val_dataset
    print("The train dataset consists of {} molecules".format(len(train_dataset)))
    print("The validation dataset consists of {} molecules".format(len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)

    model = XTBNN(len(train_dataset[0][0]), INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE,  WIDTH, FUNNEL_SIZE, DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, amsgrad = AMSGRAD)
    loss_fn = nn.MSELoss()

    Tloss, Vloss = [], []
    # training
    for epoch in range(EPOCHS):
        print("Epoch : {}".format(epoch))
        print("--------------------------")
        tloss = train(model, train_dataloader, loss_fn, optimizer)
        Tloss.append(tloss.cpu().detach().numpy())
        print("Epoch {} training loss : {}".format(epoch, tloss))
        if tloss.cpu().detach().numpy() == min(Tloss):
            torch.save({ "epoch" : epoch,
                        "state_dict" : model.state_dict}, "model_min_T.pth")
        vloss = validate(model, val_dataloader, loss_fn, optimizer)
        Vloss.append(vloss)
        print("Epoch {} validation loss : {}".format(epoch, tloss))
        if vloss == min(Vloss):
            torch.save({ "epoch" : epoch,
                        "state_dict" : model.state_dict}, "model_min_V.pth")

    model.eval()

    with open("xtbtrainout.txt", "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(train_dataloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                outfile.write(",".join(map(str, i))+"\n")
          
    with open("xtbvalout.txt", "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(val_dataloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                outfile.write(",".join(map(str, i))+"\n")
    
    with open("losses.txt", "w") as outfile:
        for i in zip(Tloss, Vloss):
            outfile.write(",".join(map(str, i))+"\n")
    
    val_target, val_pred = [], []
    train_target, train_pred = [], []

    with open("xtbvalout.txt", "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            val_target.append(float(buffer[1]))
            val_pred.append(float(buffer[2]))
    
    with open("xtbtrainout.txt", "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            train_target.append(float(buffer[1]))
            train_pred.append(float(buffer[2]))
    
    train_mae.append(mean_absolute_error(np.array(train_pred), np.array(train_target)))
    val_mae.append(mean_absolute_error(np.array(val_pred), np.array(val_target)))
    train_rmse.append(mean_squared_error(np.array(train_pred), np.array(train_target)))
    val_rmse.append(mean_squared_error(np.array(val_pred), np.array(val_target)))
    
    plt.plot(Tloss, label = "Train")
    plt.plot(Vloss, label = "Val")
    plt.legend(loc = "upper right")
    plt.savefig('losses.png')

    plt.figure(figsize = (20,10))
    plt.title("target vs prediction on test")
    plt.ylabel("Target values")
    plt.xlabel("Predicted values")
    plt.scatter(train_pred, train_target, color="blue", label="data points", s=40)
    plt.scatter(val_pred, val_target, color="red", label="data points", s=40)
    plt.legend(["Train", "Validation"])
    # for i in range(len(val_pred)):
    #     plt.annotate(mat_ids[i], (val_pred[i], val_target[i] + 0.3))
    # for i in range(len(train_pred)):
    #     plt.annotate(mat_ids[i], (train_pred[i], train_target[i] + 0.3))
    plt.plot([min(val_target), max(val_target)], [min(val_target), max(val_target)], color = "red", label = "Reference line")
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
    print("--------- RMSE----------")
    print("Train rmse: {}".format(train_rmse))
    print("Validation rmse: {}".format(val_rmse))

class XTBNN(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        width=WIDTH,
        funnel_size=FUNNEL_SIZE,
        dropout = DROPOUT_RATE
    ):
        super(XTBNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(output_channels, output_channels * 2, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.linear_stack = nn.Sequential(
            nn.Linear( (input_size - 4), width),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(width, width // funnel_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(width // funnel_size, width // (funnel_size * funnel_size)),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(width // (funnel_size * funnel_size), 1)
        )

        # # Initialize the linear layers
        # for layer in self.linear_stack:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.normal_(layer.weight, mean = 0, std = 0.09)
        #         torch.nn.init.zeros_(layer.bias)
        

    def forward(self, x):
        x = x.view(x.size(0), INPUT_CHANNELS, INPUT_SIZE)
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=1)
        # Flatten the features
        x = x.view(x.size(0), -1)
        x = self.linear_stack(x)
        return x




# function for training     
def train(model, dataloader, loss_fn, optim):
    nbatch = len(dataloader)
    model.train()
    for batch_idx, (features, targets, idx) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        train_preds = model(features)
        train_loss = loss_fn(targets, train_preds)
        optim.zero_grad()
        train_loss.backward()
        optim.step()

    return train_loss

def validate(model, dataloader, loss_fn, optim):
    nbatch = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (features, targets, idx) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            val_preds = model(features)
            val_loss_batch = loss_fn(targets, val_preds)
            val_loss += val_loss_batch.item()
    
    return val_loss/nbatch

def test(model, dataloader, loss_fn, optim):
    nbatch = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            test_preds = model(features)
            test_loss_batch = loss_fn(targets, val_preds)
            test_loss += test_loss_batch.item()
    
    return test_loss/nbatch

if __name__ == "__main__":
    print(os.listdir(DATA_DIR))
    crossval(1)
        
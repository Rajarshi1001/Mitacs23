import pymatgen.core.periodic_table
import torch
import torch.nn as nn
import sys
import re
import shutil
import os
import json
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
# data loading functions

pattern = r"^run_\d+$"
random.seed(42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SPLIT = 0.75
SEED = 42
EPOCHS = 5000
LEARNING_RATE = 0.009801507765569738
DROPOUT_RATE = 0.0233
WEIGHT_DECAY = 0.0000737688059438052
BATCH_SIZE = 128
AMS = False
FUNNEL_SIZE = 4
WIDTH = 512
KERNEL_SIZE = 3
STRIDE = 1
PADDING="same"
INPUT_SIZE = 49
IN_CHANNELS = 1
INT_CHANNELS = 100
OUT_CHANNELS = 200

train_losses, val_losses = [], []
train_mae, val_mae = [], []
train_mse, val_mse = [], []
train_mat_ids, val_mat_ids = [], []


# returning the atomic number of the species
def get_num(specie):
    try:
        return specie.number
    except AttributeError:
        if isinstance(specie, pymatgen.core.periodic_table.DummySpecies):
            try:
                return - pymatgen.core.periodic_table.Specie(specie.symbol[2:]).number
            except:
                return -1

def main(argv, num):
    ari = json.load(open("atom_init.json", "r"))
    train_database, val_database = [], []

    # train
    with open(sys.argv[1], "r") as props:
        for line in props:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            train_mat_ids.append(buffer[0])
            features = []
            for i in atoms:
                atom = pymatgen.core.periodic_table.get_el_sp(i)
                if get_num(atom) <= 92:
                    features.extend(ari[str(get_num(atom))])
                else: 
                    features.extend(ari[str(-1)])
            features.extend(bondlen)
            train_database.append([np.float32(features), np.float32([buffer[1]]), buffer[0]])
    
    # validation
    with open(sys.argv[2], "r") as props:
        for line in props:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            val_mat_ids.append(buffer[0])
            features = []
            for i in atoms:
                atom = pymatgen.core.periodic_table.get_el_sp(i)
                if get_num(atom) <= 92:
                    features.extend(ari[str(get_num(atom))])
                else: 
                    features.extend(ari[str(-1)])
            features.extend(bondlen)
            val_database.append([np.float32(features), np.float32([buffer[1]]), buffer[0]])
            
    # print("Running over {} data points".format(len(database)))
    print("The train dataset consists of {} molecules".format(len(train_database)))
    print("The validation dataset consists of {} molecules".format(len(val_database)))
    # random.shuffle(database)
    loss_fn = nn.MSELoss()
    model = NNET(len(train_database[0][0]), WIDTH, IN_CHANNELS, INT_CHANNELS, OUT_CHANNELS, FUNNEL_SIZE, DROPOUT_RATE, KERNEL_SIZE, PADDING, STRIDE).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, eps = 1e-08, weight_decay = WEIGHT_DECAY, betas =(0.9, 0.99), amsgrad = False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad = False)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr = LEARNING_RATE, alpha = 0.99, weight_decay = WEIGHT_DECAY, eps=1e-05, momentum = 0.9)
    # trainloader = DataLoader(database[:int(len(database) * SPLIT)], batch_size = BATCH_SIZE)
    # valloader = DataLoader(database[int(len(database) * SPLIT):], batch_size = BATCH_SIZE)
    trainloader = DataLoader(train_database, batch_size =BATCH_SIZE)
    valloader = DataLoader(val_database, batch_size = BATCH_SIZE)
    
    loss_T, loss_V = [], []

    MODEL_DIR = "run_{}".format(num)
    os.makedirs(MODEL_DIR) # creating a separate dir for every run of the linearNet 
    
    for i in range(EPOCHS):
        print("EPOCH {}:".format(i))
        tloss = train(trainloader, model, loss_fn, optimizer)
        loss_T.append(tloss.cpu().detach().numpy())
        print("Epoch {} training loss: {}".format(i,tloss))
        if tloss.cpu().detach().numpy() == min(loss_T):
            torch.save({"epoch": i,
                        "state_dict": model.state_dict()}, os.path.join(MODEL_DIR,"model_min_T.pth"))
        vloss = test(valloader, model, loss_fn)
        loss_V.append(vloss)
        print("Epoch {} validation loss: {}".format(i,vloss))
        if vloss == min(loss_V):
            torch.save({"epoch": i,
                        "state_dict": model.state_dict()}, os.path.join(MODEL_DIR,"model_min_V.pth"))
    
    weights_path = os.path.join(MODEL_DIR, "model.pth")
    torch.save(model.state_dict(), weights_path)
    
    model.eval()
    
    with open(os.path.join(MODEL_DIR,"trainout_{}.txt".format(num)), "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(trainloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                outfile.write(",".join(map(str, i))+"\n")
          
    with open(os.path.join(MODEL_DIR,"valout_{}.txt".format(num)), "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(valloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                outfile.write(",".join(map(str, i))+"\n")
    
    with open(os.path.join(MODEL_DIR,"losses_{}.txt".format(num)), "w") as outfile:
        for i in zip(loss_T, loss_V):
            outfile.write(",".join(map(str, i))+"\n")
    
    val_target, val_pred = [], []
    train_target, train_pred = [], []

    with open(os.path.join(MODEL_DIR,"valout_{}.txt".format(num)), "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            val_target.append(float(buffer[1]))
            val_pred.append(float(buffer[2]))
    
    with open(os.path.join(MODEL_DIR,"trainout_{}.txt".format(num)), "r") as file:
        lines = file.readlines()
        for buffer in lines:
            buffer = buffer.strip().split(",")
            train_target.append(float(buffer[1]))
            train_pred.append(float(buffer[2]))
    
    train_mae.append(mean_absolute_error(np.array(train_pred), np.array(train_target)))
    val_mae.append(mean_absolute_error(np.array(val_pred), np.array(val_target)))
    train_mse.append(mean_squared_error(np.array(train_pred), np.array(train_target)))
    val_mse.append(mean_squared_error(np.array(val_pred), np.array(val_target)))
    
    plt.figure(figsize = (10,7))
    plt.plot(loss_T, label = "Train")
    plt.plot(loss_V, label = "Val")
    plt.legend(loc = "upper right")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR,"losses_{}.png".format(num)))
    plt.close()

    plt.figure(figsize = (10,7))
    plt.title("target vs prediction on test")
    plt.xlabel("Target values")
    plt.ylabel("Predicted values")
    plt.scatter(train_target, train_pred, color="blue", label="data points", s=40)
    for i in range(len(train_pred)):
        plt.annotate(train_mat_ids[i], (train_target[i], train_pred[i] + 0.3))
    plt.scatter(val_target, val_pred, color="red", label="data points", s=40)
    for i in range(len(val_pred)):
        plt.annotate(val_mat_ids[i], (val_target[i], val_pred[i] + 0.3))
    plt.legend(["Train", "Validation"])
    plt.plot([min(val_target), max(val_target)], [min(val_target), max(val_target)], color = "red", label = "Reference line")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR,"prediction_{}.png".format(num)))
    plt.close()
    print("datapoints: {}".format(len(val_pred + train_pred)))

def crossval(n):
    print("Training across {} sets".format(n))
    for folder in os.listdir():
        if re.match(pattern, folder):
            shutil.rmtree(folder)
    for i in range(n):
        main(sys.argv, i)
    print("---------- MAE ----------")
    print("Train mae: {}".format(train_mae))
    print("Validation mae: {}".format(val_mae))
    print("--------- MSE----------")
    print("Train mse: {}".format(train_mse))
    print("Validation mse: {}".format(val_mse))

class NNET(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE, 
        width=512, 
        in_channels = IN_CHANNELS,
        int_channels = INT_CHANNELS,
        out_channels = OUT_CHANNELS,
        funnel=FUNNEL_SIZE, 
        dropout=DROPOUT_RATE,
        kernel_size=KERNEL_SIZE,
        padding="same",
        stride=STRIDE
    ):
        super().__init__()
        
        padding_size = (kernel_size - 1) // 2 if padding == "same" else 0
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding_size)
        self.embedding1 = nn.Linear(input_size//2,int_channels)
        self.embedding2 = nn.Linear(int_channels, out_channels)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(out_channels, width),
            # nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.GELU(),
            nn.Linear(width//funnel, width//funnel//funnel),
            nn.GELU(),
            nn.Linear(width//funnel//funnel, 1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling layer
        
        # Initialize the linear layers
        # for layer in self.linear_relu_stack:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.normal_(layer.weight, mean = 0, std = 0.09)
        #         torch.nn.init.zeros_(layer.bias)
        
        # # Initialize the batch normalization layers
        # for layer in self.linear_relu_stack:
        #     if isinstance(layer, nn.BatchNorm1d):
        #         torch.nn.init.constant_(layer.weight, 1)
        #         torch.nn.init.constant_(layer.bias, 0)

     
    # old approach
    def forward(self, x):
        shortcut = x
        # for _ in range(2):
        x1, x2, x3 = torch.split(x, split_size_or_sections=[24, 24, 1], dim=1)
        x1, x2 = x1.view(x1.shape[0], 24), x2.view(x2.shape[0], 24)
        x1 = self.embedding2(self.embedding1(x1)).unsqueeze(1)
        x2 = self.embedding2(self.embedding1(x2)).unsqueeze(1)
        x = torch.cat((x1, x2), dim=1)
        # print(x.shape)
        x = self.global_avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.squeeze(1)
        x = self.linear_relu_stack(x)

        return x


def train(dataloader, model, lossfn, optimizer):
    model.train()
    for batch, (in_fea, target, ids) in enumerate(dataloader):
        x = in_fea.to(device)
        y = target.to(device)
        pred = model(x)
        loss = lossfn(pred, y)
        
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.5)
        loss.backward()
        optimizer.step()
    return loss

def test(dataloader, model, lossfn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (in_fea, target, ids) in enumerate(dataloader):
            x = in_fea.to(device)
            y = target.to(device)
            pred = model(x)
            test_loss += lossfn(pred, y).item()
    return test_loss / len(dataloader)


if __name__ == "__main__":
    # executing the script
    crossval(1)



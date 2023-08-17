import pymatgen.core.periodic_table
import torch
import torch.nn as nn
import sys
import json
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import wandb
import os
import shutil
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from elementembeddings.core import Embedding
from elementembeddings.composition import CompositionalEmbedding

def useEmbeddings():

    # magpie = Embedding.load_data("magpie")
    with open(sys.argv[1], "r") as idprop:
        for line in idprop:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            train_mat_ids.append(buffer[0])
            molecule = atoms[0] + atoms[1]
            features = CompositionalEmbedding(formula = molecule, embedding = "mat2vec")
            features = np.array(features.feature_vector()[:50])
            features = (features - np.min(features))/(np.max(features) - np.min(features))
            train_database.append([np.float32(features), np.float32([buffer[1]]), buffer[0]])

    with open(sys.argv[2], "r") as idprop:
        for line in idprop:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            val_mat_ids.append(buffer[0])
            molecule = atoms[0] + atoms[1]
            features = CompositionalEmbedding(formula = molecule, embedding = "mat2vec")
            features = np.array(features.feature_vector()[:50])
            features = (features - np.min(features))/(np.max(features) - np.min(features))
            val_database.append([np.float32(features), np.float32([buffer[1]]), buffer[0]])

    print("Using {} dimensional magpie representation".format(len(train_database[0][0])))
    
    return train_database, val_database

def useAtominit():

    ari = json.load(open('atom_init.json', 'r'))

    train_database, val_database = [], []

    with open(sys.argv[1], "r") as idprop:
        for line in idprop:
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
            train_database.append([np.float32(features),np.float32([buffer[1]]), buffer[0]])

    with open(sys.argv[2], "r") as idprop:
        for line in idprop:
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
            val_database.append([np.float32(features),np.float32([buffer[1]]), buffer[0]])

    return train_database, val_database


pattern = r"^run_\d+$"

random.seed(123)
#torch.manual_seed(123)

train_database, val_database = [], []
train_mae, val_mae = [], []
train_mse, val_mse = [], []
train_mat_ids, val_mat_ids = [], []

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
#device = "cpu"

def main(argv, num):
    #wandb.init()
    argv = sys.argv
    print(argv)
    #load atominit
    batch_size = 128 #wandb.config.batch_size
    lr = 0.009801507765569738 #wandb.config.lr
    wd = 0.0009688059438052 #wandb.config.wd
    droprate = 0.0104091024895331954 #wandb.config.dropout
    nn_width = 512 #wandb.config.nn_width
    epochs = 5000 #wandb.config.epochs
    funnel = 4 #wandb.config.funnel
    tvsplit = 0.8
    ams = False #wandb.config.ams

    print(f"Using {device} device")

    train_database, val_database =  [], [] 
    train_database, val_database = useEmbeddings()
    # print(database[0])
    
    model = SimpleNetwork(len(train_database[0][0]), droprate, nn_width, funnel).to(device)
    #print(model)
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad = ams)
    
    # random.shuffle(database)
    print("The train dataset consists of {} molecules".format(len(train_database)))
    print("The validation dataset consists of {} molecules".format(len(val_database)))
    trainloader = DataLoader(train_database, batch_size = batch_size)
    valloader = DataLoader(val_database, batch_size = batch_size)
    
    loss_T = []
    loss_V = []

    MODEL_DIR = "run_{}".format(num)
    os.makedirs(MODEL_DIR) # creating a separate dir for every run of the linearNet 
    
    for i in range(epochs):
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
    plt.scatter(val_target, val_pred, color="red", label="data points", s=40)
    plt.legend(["Train", "Validation"])
    for i in range(len(val_pred)):
        plt.annotate(val_mat_ids[i], (val_target[i], val_pred[i] + 0.3))
    for i in range(len(train_pred)):
        plt.annotate(train_mat_ids[i], (train_target[i], train_pred[i] + 0.3))
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

def get_num(specie):
    try:
        return specie.number
    except AttributeError:
        if isinstance(specie, pymatgen.core.periodic_table.DummySpecies):
            try:
                return - pymatgen.core.periodic_table.Specie(specie.symbol[2:]).number
            except:
                return -1


class SimpleNetwork(nn.Module):
    def __init__(self, insize = 44, dropout = 0.15, width = 256, funnel = 1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(insize, width),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            # nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width//funnel, width//funnel//funnel),
            # nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width//funnel//funnel, width//funnel//funnel//funnel),
            nn.GELU(),
            nn.Linear(width//funnel//funnel//funnel, 1)
        )

    def forward(self, x):
        # print(x.shape)
        logits = self.linear_relu_stack(x)
        # print(logits.shape)
        return logits

def train(dataloader, model, lossfn, optimizer):
    nbatch = len(dataloader)
    model.train()
    for batch, (in_fea, target, ids) in enumerate(dataloader):
        #print(in_fea)
        x = in_fea.to(device)
        y = target.to(device)
        #print("{}, {}, {}".format(batch, target, ids))
        pred = model(x)
        #print(pred)
        loss = lossfn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
    return loss
    
def test(dataloader, model, lossfn):
    nbatch = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (in_fea, target, ids) in enumerate(dataloader):
            #print(target)
            x = in_fea.to(device)
            y = target.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            test_loss += lossfn(pred, y).item()
    return test_loss/nbatch
    

if __name__ == "__main__":

    crossval(2)

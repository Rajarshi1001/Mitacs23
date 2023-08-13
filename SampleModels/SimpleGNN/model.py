import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import shutil
import re, sys
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from elementembeddings.core import Embedding
from elementembeddings.composition import CompositionalEmbedding


pattern = r"^run_\d+$"

random.seed(123)
#torch.manual_seed(123)

train_database, val_database = [], []
train_mae, val_mae = [], []
train_mse, val_mse = [], []
train_mat_ids, val_mat_ids = [], []
mat_ids = []

device = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backend.mps.is_available()
    else "cpu"
)

LEARNING_RATE = 0.009801507765569738
BATCH_SIZE = 128
EPOCHS = 5000
WIDTH = 1024
FUNNEL_SIZE = 8 
FEATURE_LEN = 25
DROPOUT = 0.0104091024895331954
NUM_CONV = 2
AMSGRAD = False
WEIGHT_DECAY = 0.0007688059438052

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

def main(argv, num):

    train_database, val_database = useEmbeddings()
    # print(database[0])
    model = AtomPoolGNN(len(train_database[0][0]), FEATURE_LEN, NUM_CONV, WIDTH, FUNNEL_SIZE, DROPOUT).to(device)
    #print(model)
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad = AMSGRAD)
    
    # random.shuffle(database)
    print("The train dataset consists of {} molecules".format(len(train_database)))
    print("The validation dataset consists of {} molecules".format(len(val_database)))
    trainloader = DataLoader(train_database, batch_size = BATCH_SIZE, shuffle = True)
    valloader = DataLoader(val_database, batch_size = BATCH_SIZE, shuffle = True)

    loss_T = []
    loss_V = []

    MODEL_DIR = "run_{}".format(num)
    os.makedirs(MODEL_DIR) # creating a separate dir for every run of the linearNet 

    log_dir = os.path.join(MODEL_DIR, "logs")
    writer = SummaryWriter(log_dir)
    
    for i in range(EPOCHS):
        print("EPOCH {}:".format(i))
        tloss = train(trainloader, model, loss_fn, optimizer)
        loss_T.append(tloss.cpu().detach().numpy())
        writer.add_scalar("Loss/Train", tloss, i)
        print("Epoch {} training loss: {}".format(i,tloss))
        if tloss.cpu().detach().numpy() == min(loss_T):
            torch.save({"epoch": i,
                        "state_dict": model.state_dict()}, os.path.join(MODEL_DIR,"model_min_T.pth"))
        vloss = test(valloader, model, loss_fn)
        writer.add_scalar("Loss/Val", vloss, i)
        loss_V.append(vloss)
        print("Epoch {} validation loss: {}".format(i,vloss))
        if vloss == min(loss_V):
            torch.save({"epoch": i,
                        "state_dict": model.state_dict()}, os.path.join(MODEL_DIR,"model_min_V.pth"))
    writer.close()
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

# Model arch
class ReConvLayer(nn.Module):
    def __init__(self, insize, feature_len):

        super(ReConvLayer,self).__init__()        
        self.convfc = nn.Linear(insize, feature_len*2)
        self.convsigmoid = nn.Sigmoid()
        self.convsoftplus1 = nn.Softplus()
        self.convsoftplus2 = nn.Softplus()
        self.convbn1 = nn.BatchNorm1d(feature_len*2)
        self.convbn2 = nn.BatchNorm1d(feature_len)

    def forward(self, atom1, atom2):

        atom1_g = self.convfc(torch.cat((atom1, atom2),dim =1))
        atom1_g = self.convbn1(atom1_g)
        atom1_filter, atom1_core = atom1_g.chunk(2, dim = 1)
        atom1_filter = self.convsigmoid(atom1_filter)
        atom1_core = self.convsoftplus1(atom1_core)
        atom1_sumed = atom1_filter * atom1_core
        atom1_sumed = self.convbn2(atom1_sumed)
        atom1_g = self.convsoftplus2(atom1 + atom1_sumed)
        
        atom2_g = self.convfc(torch.cat((atom2, atom1), dim=1))
        atom2_g = self.convbn1(atom2_g)
        atom2_filter, atom2_core = atom2_g.chunk(2, dim = 1)
        atom2_filter = self.convsigmoid(atom2_filter)
        atom2_core = self.convsoftplus1(atom2_core)
        atom2_sumed = atom2_filter * atom2_core
        atom2_sumed = self.convbn2(atom2_sumed)
        atom2_g = self.convsoftplus2(atom2 + atom2_sumed)
        
        return atom1_g, atom2_g

class AtomPoolGNN(nn.Module):

    def __init__(self, insize = 26, feature_len = FEATURE_LEN, nconv = NUM_CONV, width = WIDTH, funnel = FUNNEL_SIZE, dropout = DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList([ReConvLayer(insize, feature_len) for i in range(nconv)])     
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_len, width),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.GELU(),
            nn.Linear(width//funnel, width//funnel//funnel),
            nn.GELU(),
            nn.Linear(width//funnel//funnel, 1)
        )
        self.feature_len = feature_len

    def forward(self, x):
        features = torch.split(x, self.feature_len, dim = 1)
        # assert(len(features) > 2)
        atom1 = features[0]
        atom2 = features[1]
        # globals = features[2]
        for conv_func in self.convs:
            atom1, atom2 = conv_func(atom1, atom2)
        atoms_pooled = torch.mean(torch.stack((atom1,atom2)), dim = 0)
        logits = self.linear_relu_stack(atoms_pooled)
        return logits
        
    def init_weights(self, initfunct, *args, **kwargs):

        self.initfunct = initfunct
        self.apply(lambda m: self._init_weights(m, *args, **kwargs))
        
    def _init_weights(self, module, *args, **kwargs):

        if isinstance(module, nn.Linear):
            self.initfunct(module.weight, *args, **kwargs)
            if module.bias is not None:
                module.bias.data.zero_

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
    crossval(5)
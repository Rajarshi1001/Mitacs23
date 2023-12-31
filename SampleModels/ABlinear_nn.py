import pymatgen.core.periodic_table
import torch
import torch.nn as nn
import sys
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
#import wandb
import argparse 
import os
import time
import json
import GPUtil
import lib.models as libmod

random.seed(567)

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
#device = "cpu"
torch.cuda.set_device(GPUtil.getAvailable("load")[0])

def main():
    ##wandb stuff
    #wandb.init()
    argv = sys.argv #wandb.config.argv
    print(argv)
    #batch_size = 1000 #wandb.config.batch_size
    #lr = 1e-4 #wandb.config.lr
    #wd = 1e-6 #wandb.config.wd
    #droprate = 0.1 #wandb.config.dropout
    #nn_width = 2048 #wandb.config.nn_width
    #epochs = 50 #wandb.config.epochs
    #funnel = 8 #wandb.config.funnel
    #tvsplit = 0.8
    ams = False #wandb.config.ams
    log = False
    
    #parse args
    parser = argparse.ArgumentParser(description='AB x Linear')
    parser.add_argument('root_dir', metavar='OPTIONS', help='path to root dir, '
                         'then other options')
    parser.add_argument('--id-prop-v', metavar='N', default='id_clean_v.csv', type=str, help="Cif list (csv) Validation")
    parser.add_argument('--id-prop-t', "-t", metavar='N', default='id_clean_t.csv', type=str, help="Cif list (csv) Testing")
    parser.add_argument("--out", metavar = "N", default="outputs", type = str, help = "output folder location")
    parser.add_argument("-e", metavar = "N", default = 5000, type = int, help = "epochs")
    parser.add_argument("-d", metavar = "N", default = 0.00, type = float, help = "dropout")
    parser.add_argument("--width", metavar = "N", default = 1024, type = int, help = "netwidth")
    parser.add_argument("--funnel", metavar = "N", default = 8, type = int, help = "funnel rate")
    parser.add_argument("--lr", metavar = "N", default = 0.009801507765569738, type = float, help = "lr")
    parser.add_argument("--wd", metavar = "N", default = 0.0007688059438052, type = float, help = "wd")
    parser.add_argument("-b", metavar = "N", default = 1000, type = int, help = "batch size")
    parser.add_argument("-m", metavar = "N", default = 0, type = int, help = "model type: 0: linear, 1: +pooling")
    parser.add_argument("-c", metavar = "N", default = 3, type = int, help = "conv layer count")
    args, unknowns = parser.parse_known_args(sys.argv[1:])
    
    epochs = args.e
    droprate = args.d
    nn_width = args.width
    funnel = args.funnel
    lr = args.lr
    wd = args.wd
    batch_size = args.b
    nconv = args.c
    
    prefix = args.root_dir
    train_file = prefix + "/" + args.id_prop_t
    val_file = prefix + "/" + args.id_prop_v
    #load atominit
    print("Using {} device {}".format(device, GPUtil.getAvailable("load")[0]))
    outfold = "./train_outputs/{}".format(args.out)
    model_file_v = outfold + "/" + "model_min_V.pth"
    model_file_t = outfold + "/" + "model_min_T.pth"
    train_ofile = outfold + "/" + "trainout.csv"
    val_ofile = outfold + "/" + "valout.csv"
    val_pred_ofile = outfold + "/" + "valpred.csv"
    
    if not os.path.exists(outfold):
        os.makedirs(outfold)
    
    ari = json.load(open('atom_init.json', 'r'))
    arilen = len(ari["1"])
    
    database_t = []
    with open(train_file, "r") as idprop:
        for line in idprop:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            features = []
            for i in atoms:
                atom = pymatgen.core.periodic_table.get_el_sp(i)
                features.extend(ari[str(get_num(atom))])
            #database[buffer[0]] = np.array(features)
            #id_prop[buffer[0]] = buffer[1]
            features.extend(bondlen)
            target = np.float32([buffer[1]])
            if log:
                target = np.log(target + 1)
            database_t.append([np.float32(features), target, buffer[0]])
    database_v = []
    with open(val_file, "r") as idprop:
        for line in idprop:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            bondlen = buffer[4:]
            features = []
            for i in atoms:
                atom = pymatgen.core.periodic_table.get_el_sp(i)
                features.extend(ari[str(get_num(atom))])
            #database[buffer[0]] = np.array(features)
            #id_prop[buffer[0]] = buffer[1]
            features.extend(bondlen)
            target = np.float32([buffer[1]])
            if log:
                target = np.log(target + 1)
            database_v.append([np.float32(features), target, buffer[0]])
    
    if args.m == 0:
        model = libmod.SimpleNetwork(len(database_t[0][0]), droprate, nn_width, funnel).to(device)
    elif args.m == 1:
        model = libmod.PoolNetwork(len(database_t[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 2:
        model = libmod.PoolConvNetwork(len(database_t[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 3:
        model = libmod.BilinNetwork(len(database_t[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 4:
        model = libmod.SimpleGNN(len(database_t[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    elif args.m == 5:
        model = libmod.AtomPoolGNN(len(database_t[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    else:
        print("model type not found")
        return 1
    #print(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad = ams)
    
    random.shuffle(database_t)
    random.shuffle(database_v)
    
    #targtetlist = [i[1] for i in (database_t + database_v).random.sample(len(database_t))]
    #normalizer = Normalizer(targetlist)
    
    trainloader = DataLoader(database_t, batch_size = batch_size, shuffle = True)
    valloader = DataLoader(database_v, batch_size = batch_size, shuffle = True)
    
    loss_T = []
    loss_V = []
    tstart = time.time()
    
    for i in range(epochs):
        print("EPOCH {} load time: {}".format(i, time.time() - tstart))
        tstart = time.time()
        tloss = train(trainloader, model, loss_fn, optimizer)
        loss_T.append(tloss.cpu().detach().numpy())
        print("Epoch {} training loss: {}".format(i,tloss), end = "")
        if tloss.cpu().detach().numpy() == min(loss_T):
            #train_best = model.state_dict().copy()
            torch.save(model.state_dict(), model_file_t)
            print("*")
        else:
            print("")
        vloss = test(valloader, model, loss_fn)
        loss_V.append(vloss)
        print("Epoch {} validation loss: {}".format(i,vloss), end = "")
        if vloss == min(loss_V):
            #val_best = model.state_dict().copy()
            torch.save(model.state_dict(), model_file_v)
            print("*")
        else:
            print("")
        # wandb.log({
            # "epoch": i,
            # "train_loss": tloss,
            # "val_loss": vloss
        # })
    #torch.save(train_best, model_file_t)
    #torch.save(val_best, model_file_v)
    model.eval()
    
    with open(train_ofile, "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(trainloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            if log:
                for i in zip(ids, (np.exp(target[:,0].detach()) - 1).tolist(), (np.exp(pred[:,0].detach().tolist()) - 1)):
                    outfile.write(",".join(map(str, i))+"\n")
            else:
                for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                    outfile.write(",".join(map(str, i))+"\n")
          
    with open(val_ofile, "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(valloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            if log:
                for i in zip(ids, (np.exp(target[:,0].detach()) - 1).tolist(), (np.exp(pred[:,0].detach().tolist()) - 1)):
                    outfile.write(",".join(map(str, i))+"\n")
            else:
                for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                    outfile.write(",".join(map(str, i))+"\n")
    
    with open(outfold + "/losses.csv", "w") as outfile:
        for i in zip(loss_T, loss_V):
            outfile.write(",".join(map(str, i))+"\n")
    
    plt.plot(loss_T, label = "Train")
    plt.plot(loss_V, label = "Val")
    plt.legend(loc = "upper right")
    plt.ylim([0,1])
    plt.title("lr {:.3E}, wd {:.3E}, dropout {:.3E},\nwidth {}, funnel {}, batchsize {}".format(lr, wd, droprate, nn_width, funnel, batch_size))
    plt.savefig(outfold + '/losses.png')
    
    model.load_state_dict(torch.load(model_file_v))
    model.eval()
          
    with open(val_pred_ofile, "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(valloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            if log:
                for i in zip(ids, (np.exp(target[:,0].detach()) - 1).tolist(), (np.exp(pred[:,0].detach().tolist()) - 1)):
                    outfile.write(",".join(map(str, i))+"\n")
            else:
                for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                    outfile.write(",".join(map(str, i))+"\n")


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4)
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
    
def get_num(specie):
    try:
        return specie.number
    except AttributeError:
        if isinstance(specie, pymatgen.core.periodic_table.DummySpecies):
            try:
                return - pymatgen.core.periodic_table.Specie(specie.symbol[2:]).number
            except:
                return -1

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

# sweep_configuration = {
    # 'method': 'bayes',
    # 'name': 'Bayesian Search 5 val',
    # 'metric': {'goal': 'minimize', 'name': 'val_loss'},
    # 'parameters': 
    # {
        # 'batch_size': {'values': [5, 10, 20, 40, 80]},
        # #'epochs': {'values': [1000, 2000, 5000]},
        # 'lr': {'max': 0.001, 'min': 0.000001},
        # 'wd': {'max': 0.00001, 'min': 0.000001},
        # 'dropout': {'max': 0.1, 'min': 0.0},
        # #'nn_width': {'values': [128, 256, 512, 1024]},
        # #"funnel": {'values': [8]},
        # "argv":{"values": [sys.argv]},
        # "ams":{"values": [True, False]}
    # },
    # "program": "ABlinear_nn.py",
# }

# sweep_id = wandb.sweep(
  # sweep=sweep_configuration, 
  # project='AB Linear Networks'
  # )

#wandb.agent(sweep_id, function = main)
main()

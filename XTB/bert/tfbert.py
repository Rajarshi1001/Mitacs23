import transformers
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import re, shutil
import random
import sys

INPUT_SIZE = 2048
REDUCED_SIZE = 512
LEARNING_RATE = 0.0098
WEIGHT_DECAY = 0.0007824
AMSGRAD = False
BATCH_SIZE = 16
HIDDEN_SIZE = 768
OUTPUT_SIZE = 1
EPOCHS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = os.getcwd()
pattern = r"^bert_run_\d+$"
DATA_DIR = os.path.join(os.path.abspath(os.path.join(PATH, os.pardir)), "XTBData")

train_mae, val_mae = [], []
train_mse, val_mse = [], []
train_dataset, val_dataset, test_dataset = [], [], []


# defining the fine-tuned model
bert_config = BertConfig.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", config=bert_config).to(device)
linear1 = nn.Linear(INPUT_SIZE, REDUCED_SIZE).to(device)
linear2 = nn.Linear(REDUCED_SIZE*768, REDUCED_SIZE).to(device)
linear3 = nn.Linear(REDUCED_SIZE, OUTPUT_SIZE).to(device)

def preprocess():
    train_path = os.path.join(DATA_DIR, sys.argv[1])
    val_path = os.path.join(DATA_DIR, sys.argv[2])
    # test_path = os.path.join(DATA_DIR, sys.argv[3])

    # reading the train file
    train_data = pd.read_csv(train_path, header=0)
    val_data = pd.read_csv(val_path, header=0)
    # test_data = pd.read_csv(test_path, header=0)
    feat_cols = train_data.columns[2:]

    with open(train_path, "r") as train_file:
        for line in train_file:
            buffer = line.strip().split(",")
            train_features = buffer[2:]
            train_targets = float(buffer[1])
            train_id = buffer[0]
            if train_targets > 1.0:
                train_dataset.append(
                    [np.float32(train_features), np.float32([train_targets]), str(train_id)]
                )

    with open(val_path, "r") as val_file:
        for line in val_file:
            buffer = line.strip().split(",")
            val_features = buffer[2:]
            val_targets = float(buffer[1])
            val_id = buffer[0]
            if val_targets > 1.0:
                val_dataset.append(
                    [np.float32(val_features), np.float32([val_targets]), str(val_id)]
                )

    return train_dataset, val_dataset


def main(sys, num):
    train_dataset, val_dataset = preprocess()
    print("The train dataset consists of {} molecules".format(len(train_dataset)))
    print("The validation dataset consists of {} molecules".format(len(val_dataset)))
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(bert_model.parameters())
        + list(linear1.parameters())
        + list(linear2.parameters())
        + list(linear3.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    loss_T, loss_V = [], []

    MODEL_DIR = "bert_run_{}".format(num)
    os.makedirs(MODEL_DIR, exist_ok=True)

    log_dir = os.path.join(MODEL_DIR, "logs")
    writer = SummaryWriter(log_dir)

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch))
        train_loss = train(
            bert_model, linear1, linear2, linear3, trainloader, loss_fn, optimizer
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        loss_T.append(train_loss)
        print("Epoch {} training loss: {}".format(epoch, train_loss))

        val_loss = validate(
            bert_model, linear1, linear2, linear3, valloader, loss_fn
        )
        writer.add_scalar("Loss/Val", val_loss, epoch)
        loss_V.append(val_loss)
        print("Epoch {} validation loss: {}".format(epoch, val_loss))

        if val_loss == min(loss_V):
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": bert_model.state_dict(),
                },
                os.path.join(MODEL_DIR, "model_min_V.pth"),
            )

    writer.close()

    torch.save(
        {
            "epoch": EPOCHS - 1,
            "state_dict": bert_model.state_dict(),
        },
        os.path.join(MODEL_DIR, "model.pth"),
    )

    plot_losses(loss_T, loss_V, MODEL_DIR)
    plot_predictions(valloader, MODEL_DIR)


def train(bert_model, linear1, linear2, linear3, dataloader, loss_fn, optimizer):
    nbatch = len(dataloader)
    bert_model.train()
    linear1.train()
    linear2.train()
    linear3.train()
    train_loss = 0

    for batch_idx, (train_features, target, idx) in enumerate(dataloader):
        train_features = train_features.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        # Reduce the input size from 2048 to 512 using linear layer
        reduced_features = linear1(train_features).to(device) 
        reduced_features = reduced_features.long()
        # Pass the reduced features through BERT
        bert_output = bert_model(reduced_features)
        # print(bert_output.shape)
        # Flatten the BERT output
        flattened_layer = bert_output.last_hidden_state.view(train_features.size(0), -1).to(device)
        # Pass the flattened layer through the second linear layer
        output = linear2(flattened_layer).to(device)
        output = linear3(output).to(device)
        train_loss_batch = loss_fn(target.squeeze(), output.squeeze())
        train_loss += train_loss_batch.item()
        train_loss_batch.backward()
        optimizer.step()

    return train_loss / nbatch


def validate(bert_model, linear1, linear2, linear3, dataloader, loss_fn):
    nbatch = len(dataloader)
    bert_model.eval()
    linear1.eval()
    linear2.eval()
    linear2.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (val_features, target, id) in enumerate(dataloader):
            val_features = val_features.to(device)
            target = target.to(device)
            # Reduce the input size from 2048 to 512 using linear layer
            reduced_features = linear1(val_features).to(device)
            reduced_features = reduced_features.long()
            # Pass the reduced features through BERT
            bert_output = bert_model(reduced_features)
            # Flatten the BERT output
            flattened_layer = bert_output.last_hidden_state.view(val_features.size(0), -1).to(device)
            # Pass the flattened layer through the second linear layer
            output = linear2(flattened_layer).to(device)
            output = linear3(output).to(device)

            val_loss_batch = loss_fn(output, target)
            val_loss += val_loss_batch.item()

    return val_loss / nbatch



def plot_losses(loss_T, loss_V, model_dir):
    plt.figure(figsize=(10, 7))
    plt.plot(loss_T, label="Train")
    plt.plot(loss_V, label="Val")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.savefig(os.path.join(model_dir, "losses.png"))
    plt.close()


def plot_predictions(dataloader, model_dir):
    bert_model.eval()
    linear1.eval()
    linear2.eval()

    train_targets, train_preds = [], []
    with torch.no_grad():
        for (train_features, train_target, idx) in dataloader:
            train_features = train_features.to(device)
            train_target = train_target.to(device)

            # Reduce the input size from 2048 to 512 using linear layer
            reduced_features = linear1(train_features.to(torch.float))

            # Pass the reduced features through BERT
            bert_output = bert_model(reduced_features)

            # Flatten the BERT output
            flattened_layer = bert_output.last_hidden_state.view(
                train_features.size(0), -1
            )

            # Pass the flattened layer through the second linear layer
            output = linear2(flattened_layer)
            output = linear3(output)
            train_targets.extend(train_target.squeeze().tolist())
            train_preds.extend(output.squeeze().detach().cpu().tolist())

    val_targets, val_preds = [], []
    with torch.no_grad():
        for (val_features, val_target, idx) in dataloader:
            val_features = val_features.to(device)
            val_target = val_target.to(device)

            # Reduce the input size from 2048 to 512 using linear layer
            reduced_features = linear1(val_features.to(torch.float))

            # Pass the reduced features through BERT
            bert_output = bert_model(reduced_features)

            # Flatten the BERT output
            flattened_layer = bert_output.last_hidden_state.view(
                val_features.size(0), -1
            )

            # Pass the flattened layer through the second linear layer
            output = linear2(flattened_layer)
            output = linear3(output)
            val_targets.extend(val_target.squeeze().tolist())
            val_preds.extend(output.squeeze().detach().cpu().tolist())

    train_targets = np.array(train_targets)
    train_preds = np.array(train_preds)
    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)

    plt.figure(figsize=(10, 7))
    plt.scatter(train_targets, train_preds, color="blue", s=40)
    plt.scatter(val_targets, val_preds, color="red", s=40)
    plt.plot(
        [min(val_targets), max(val_targets)],
        [min(val_targets), max(val_targets)],
        color="blue",
        linestyle="--",
        label="Reference line",
    )
    plt.xlabel("Target values")
    plt.ylabel("Predicted values")
    plt.title("Target vs Prediction")
    plt.legend(["Train", "Validation"])
    plt.grid()
    plt.savefig(os.path.join(model_dir, "predictions.png"))
    plt.close()


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


if __name__ == "__main__":
    crossval(1)

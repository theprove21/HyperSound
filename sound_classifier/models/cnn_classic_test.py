import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm

import sys


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)



        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-03, eps=1e-07, weight_decay=1e-3) #lr=0.001

        self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x

    def fit(self, train_loader, epochs, val_loader=None):
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

        labels = {'preds':[],'truth':[]}

        for epoch in range(epochs):
            self.train()

            print("\nEpoch {}/{}".format(epoch+1, epochs))

            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for step, batch in enumerate(train_loader):
                    X_batch = batch['spectrogram'].to(self.device)
                    y_batch = batch['label'].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        # forward + backward
                        outputs = self.forward(X_batch)
                        batch_loss = self.criterion(outputs, y_batch)
                        # print('outputs min max: ',outputs.min().item(), outputs.max().item())
                        # print('y_batch min max',y_batch.min().item(), y_batch.max().item())
                        # print('difference min max',(outputs - y_batch).min().item() , (outputs - y_batch).max().item())
                        batch_loss.backward()

                        # update the parameters
                        self.optimizer.step()

                    pbar.update(1)

            # model evaluation - train data
            train_loss, train_acc, train_preds, train_labels = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            labels['preds'].append(val_preds)
            lables['truth'].append(val_labels)
            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        return history, labels

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            outputs = self.forward(X)

        return outputs

    def evaluate(self, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

        all_preds = []
        all_labels = []
    
        for step, batch in enumerate(data_loader):
            X_batch = batch['spectrogram'].to(self.device)
            y_batch = batch['label'].to(self.device)
    
            outputs = self.predict(X_batch)
    
            # get batch loss
            loss = self.criterion(outputs, y_batch)
            running_loss = running_loss + loss
    
            # calculate batch accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)
    
            # collect for further analysis
            all_preds.append(predictions.cpu())
            all_labels.append(y_batch.cpu())
    
        loss = running_loss.item() / (step+1)
        accuracy = running_acc.item() / (step+1)
    
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    
        return loss, accuracy, all_preds, all_labels

def build_model(**kwargs):

    if torch.cuda.is_available():

        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)

    net = Net(device).to(device)
    print(net)

    return Net(device, **kwargs)

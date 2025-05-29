import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Net(nn.Module):
    def __init__(self, device, n_class=10):
        super(Net, self).__init__()


        self.layer1 = Conv2d(1,   64,  pooling=(2,2))
        self.layer2 = Conv2d(64, 128, pooling=(2,2))
        self.layer3 = Conv2d(128,128, pooling=(2,2))
        self.layer4 = Conv2d(128,128, pooling=(2,2))
        self.layer5 = Conv2d(128, 64, pooling=(2,2))

        self.fc = nn.Linear(64, n_class)

        self.pool = nn.AvgPool2d(kernel_size=4)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-03, eps=1e-07, weight_decay=1e-3) #lr=0.001

        self.device = device

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5)
        x = self.fc(x)

        return x

    def fit(self, train_loader, epochs, val_loader=None):
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

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
            train_loss, train_acc = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        return history

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            outputs = self.forward(X)

        return outputs

    def evaluate(self, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

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

        loss = running_loss.item() / (step+1)
        accuracy = running_acc.item() / (step+1)

        return loss, accuracy

def build_model(**kwargs):

    if torch.cuda.is_available():
    device = torch.device("cuda:0")
    else:
    device = torch.device("cpu")
    print(device)

    net = Net(device).to(device)
    print(net)

    return Net(**kwargs)
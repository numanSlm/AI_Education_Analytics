import os
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid

import PIL.Image as Image

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models


#################################################################################################################
# Dataset class
class ImageDataset(Dataset):
    # label function
    def _label_func(self, file_path):
        '''
        this function is used to extract the name/label from the path given
        '''
        return file_path.split("/")[-2]

    #Class constructor
    def __init__(self,path,):

        self.path = os.path.abspath(path)
        self.folders = os.listdir(path)

        self.files = []
        for folder in self.folders:
            files = os.listdir(os.path.join(self.path,folder))
            abspath_fils = list(map(lambda x: os.path.join(self.path,folder,x),files))
            self.files.extend(abspath_fils)

        self.labels = list(map(lambda x:self._label_func(x),self.files))

        self.transforms = transforms.Compose([
                                              transforms.Resize(size=(48, 48)),
                                              transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(p= 0.5),
                                              transforms.RandomRotation(degrees=(-10,10)),

        ])

        self.classes = list(set(self.labels))
        self.n_class = len(self.classes)
        self.classes_dict = dict()
        for idx,cls in enumerate(self.classes):
            self.classes_dict[cls] = idx
    #len function
    def __len__(self):
        return len(self.files)

    #getitem function
    def __getitem__(self, idx):

        Im=self.transforms(Image.open(self.files[idx]).convert(mode='L'))
        label = F.one_hot(T.tensor(self.classes_dict[self.labels[idx]]),self.n_class)

        return Im, label
    

######################################################################################################################################


    ######################################################################################################################################
# Dataloader class

class Learner:
    def __init__(self, train_dl, val_dl, model,labels_name, base_lr=0.001, base_wd = 0.0,save_best=None,log_path=None):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.labels_name = labels_name
        self.save_best =  save_best
        self.log_path = log_path
        #using cuda if available
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.epoch = 0
        self.base_lr = base_lr
        self.base_wd = base_wd
        #CrossEntropyLoss criterion
        self.criterion = nn.CrossEntropyLoss()
        #Adam optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=self.base_lr,weight_decay=self.base_wd)

        self.logs = []
        self.metrics = {}
        self.best_score = 0.0



################################################################################################################################################

    # Training function

    def train_one_epoch(self,lr,wd=0.0):
        """
        Trains the neural network model for one epoch using the provided learning rate and weight decay.

        Parameters:
            - lr (float): Learning rate for the optimizer.
            - wd (float, optional): Weight decay parameter for regularization (default is 0.0).

        Returns:
            - train_loss (float): Average training loss for the epoch.

        Note:
            - This function assumes that the class instance has the following attributes:
                - model: Neural network model
                - optim: Optimizer for training
                - criterion: Loss criterion
                - train_dl: Training data loader
                - device: Device (CPU or GPU) on which the model and data should be placed
                - metrics: Dictionary to store training metrics (e.g., training loss)
        """
        self.model.train()
        self.lr = lr
        self.wd=wd
        # Set learning rate
        for g in self.optim.param_groups:
            g['lr'] = self.lr
        # Set weight decay
        for g in self.optim.param_groups:
            g['weight_decay'] = self.wd

        self.model.to(self.device)
        self.model.train()
        train_loss = 0.0

        # Training loop
        for inputs, labels in tqdm(self.train_dl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optim.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.to(T.float))

            loss.backward()
            self.optim.step()

            train_loss += loss.item()
        # Compute metrics
        self.metrics["train loss"] = train_loss / len(self.train_dl)
        return train_loss
################################################################################################################################################
    def eval(self):
        """
        Evaluates the neural network model on the validation dataset and computes various classification metrics.

        Returns:
            None

        Note:
            - This function assumes that the class instance has the following attributes:
                - model: Neural network model
                - val_dl: Validation data loader
                - device: Device (CPU or GPU) on which the model and data should be placed
                - criterion: Loss criterion
                - metrics: Dictionary to store evaluation metrics (e.g., validation loss, accuracy, precision, recall, F1 score)
        """
        self.model.eval()

        self.val_loss = 0.0
        self.all_predictions = []
        self.all_labels = []

        # Validation loop
        with T.no_grad():
            for inputs, labels in tqdm(self.val_dl, position=0, leave=True):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.outputs = self.model(inputs)
                loss = self.criterion(self.outputs, labels.to(T.float))
                self.val_loss += loss.item()
                # Get predictions
                _, predictions = T.max(self.outputs, 1)

                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_labels.extend(np.argmax(labels.cpu().numpy(),axis=1))

        # Compute metrics
        # accracy calculation TP+TN/TP+TN+FP+FN
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        # precision calculation TP/TP+FP
        precision_micro = precision_score(self.all_labels, self.all_predictions, average='micro',zero_division=0)
        # recall calculation TP/TP+FN
        recall_micro = recall_score(self.all_labels, self.all_predictions, average='micro')
        # F1 score calculation 2*(precision*recall)/(precision+recall)
        f1_micro = f1_score(self.all_labels, self.all_predictions, average='micro')

        # precision calculation TP/TP+FP macro
        precision_macro = precision_score(self.all_labels, self.all_predictions, average='macro',zero_division=0)
        # recall calculation TP/TP+FN macro
        recall_macro = recall_score(self.all_labels, self.all_predictions, average='macro')
        # F1 score calculation 2*(precision*recall)/(precision+recall) macro
        f1_macro = f1_score(self.all_labels, self.all_predictions, average='macro')
        self.metrics['Validation Loss'] = self.val_loss / len(self.val_dl)

        self.metrics['Accuracy']= accuracy
        self.metrics['Precision Micro']= precision_micro
        self.metrics['Recall Micro']= recall_micro
        self.metrics['F1 Score Micro']= f1_micro
        self.metrics['Precision Macro']= precision_macro
        self.metrics['Recall Macro']= recall_macro
        self.metrics['F1 Score Macro']= f1_macro


################################################################################################################################################

    # Print metrics function

    def print_metrics(self):
        formatted_metrics = {key: f"{value:.{4}f}" if isinstance(value, (float, np.float32, np.float64)) else value
                                for key, value in self.metrics.items()}

        metric_str = ", ".join([f"{key}: {value}" for key, value in formatted_metrics.items()])
        print(metric_str)



################################################################################################################################################

    # Training and evaluation function

    def train_eval(self,lr,epochs,wd=0.0):
        for _ in range(epochs):
            self.metrics = {"epoch": self.epoch}
            self.train_one_epoch(lr,wd)
            self.eval()
            self.logs.append(self.metrics)
            self.epoch += 1
            self.print_metrics()

            # Save best model
            if self.metrics["Accuracy"] > self.best_score:
                self.best_score = self.metrics["Accuracy"]
                # Save model
                if self.save_best != None:
                    self.save(self.save_best)
                    print("\nNew best score -- model saved")
            self.save_log(self.log_path)



################################################################################################################################################

    # Save and load functions

    def save_log(self,path):
        df = pd.DataFrame(self.logs)
        df.to_csv(path + ".csv")

################################################################################################################################################
    def save(self,path):
        T.save(self.model.state_dict(), path)
        self.save_log(self.log_path)

################################################################################################################################################
    def load(self,path):
        state = T.load(path)
        self.model.load_state_dict(state_dict=state,strict=False)

################################################################################################################################################

    def load_log(self,path):
        df = pd.read_csv(path)
        self.log = df.to_json()


################################################################################################################################################

    #visualize function min max normalization

    def plot_metrics_micro(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df['epoch'],df['Accuracy'],label="Accuracy")
        plt.plot(df['epoch'],df['Precision Micro'],label="Precision Micro'")
        plt.plot(df['epoch'],df['Recall Micro'],label="Recall Micro")
        plt.plot(df['epoch'],df['F1 Score Micro'],label="F1 Score Micro")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("score")
        if save_path != None:
            plt.savefig(save_path)


################################################################################################################################################

    #visualize function min max normalization macro

    def plot_metrics_macro(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df['epoch'],df['Accuracy'],label="Accuracy")
        plt.plot(df['epoch'],df['Precision Macro'],label="Precision Macro'")
        plt.plot(df['epoch'],df['Recall Macro'],label="Recall Macro")
        plt.plot(df['epoch'],df['F1 Score Macro'],label="F1 Score Macro")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("score")
        if save_path != None:
            plt.savefig(save_path)


################################################################################################################################################
    #visualize plot loss
    def plot_loss(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df["epoch"],df["train loss"], label = "Train Loss")
        plt.plot(df["epoch"],df["Validation Loss"],label = "Validation Loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        if save_path != None:
            plt.savefig(save_path)



################################################################################################################################################
    # predict function for test set

    def predict(self, test_dl):
        if T.cuda.is_available():
            self.model.cuda()
            
        all_predictions = []
        all_labels = []
        all_probabilites = []
        self.model.eval()
        with T.no_grad():
            for inputs, labels in tqdm(test_dl, position=0, leave=True):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.outputs = self.model(inputs)

                _, predictions = T.max(self.outputs, 1)
                probabilities = T.nn.functional.softmax(self.outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(np.argmax(labels.cpu().numpy(),axis=1))
                all_probabilites.extend(probabilities.cpu().numpy())

        self.test_preds = all_predictions
        self.test_labels = all_labels
        self.test_pred_prob = all_probabilites

        return (all_predictions,all_labels, all_probabilites)
    

################################################################################################################################################

    #visualuze confusion matrix

    def plotConfisuionMatrix(self,save_path):
        cm = confusion_matrix(self.test_preds, self.test_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.labels_name)
        disp.plot(cmap=plt.cm.hot_r)
        plt.savefig(save_path)
################################################################################################################################################

# ResNet class
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ResBlock, self).__init__()
        # Residual block is composed of two convolutional layers with batch normalization and ReLU activation and a residual connection
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = kernel_size//2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = kernel_size//2),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
    # forward function of the residual block
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# ResNet class
class ResNet(nn.Module):
    # Class constructor
    def __init__(self,num_classes, num_channel, num_blocks, num_fc_layers, avg_pool_size, kernel_size = 3,):
        super(ResNet, self).__init__()
        # ResNet is composed of a convolutional layer, a residual block and a fully connected layer
        self.resblocks = self._make_resblocks(num_channel, num_blocks,kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))

        self.fc_layers = self._make_fc_layers(num_channel* avg_pool_size * avg_pool_size, num_fc_layers,num_classes)

    # Function to create residual blocks
    def _make_resblocks(self, bloc_size, num_resblocks,kernel_size):
        layers = []
        layers.append(ResBlock(1, bloc_size,kernel_size))
        for _ in range(num_resblocks-1):
            layers.append(ResBlock(bloc_size, bloc_size,kernel_size))
        return nn.Sequential(*layers)
    # Function to create fully connected layers
    def _make_fc_layers(self, fc_size, num_fc_layers,num_classes):
        layers = []
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(fc_size, fc_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_size,num_classes))
        return nn.Sequential(*layers)
    # forward function of the ResNet
    def forward(self, x):
        x = self.resblocks(x)
        x = self.avg_pool(x)
        x = T.flatten(x, 1)
        x = self.fc_layers(x)
        return x

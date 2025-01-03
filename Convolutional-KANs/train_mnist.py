import sys
sys.path.insert(1,'Convolutional-KANs')

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import KKAN_Convolutional_Network
from architectures_28x28.conv_and_kan import NormalConvsKAN_Medium
from architectures_28x28.CKAN_BN import CKAN_BN
from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.SimpleModels import *
from architectures_28x28.ConvNet import ConvNet
from evaluations import *
from hiperparam_tuning import *
from generic_train import train_model_generic

torch.manual_seed(42) #Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

print(len(mnist_train))    


dataset_name = "MNIST"
path = f"models/{dataset_name}"

if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists(path):
    os.mkdir(path)

def join_path(name,pa):
  print(os.path.join(pa,name+".pt"))
  return os.path.join(pa,name+".pt")

model_KANC_MLP= KANC_MLP_Medium()
print(f"============== Training model {model_KANC_MLP.name} ============")
train_model_generic(model_KANC_MLP, mnist_train, mnist_test,device,epochs = 10,path=path)

model_KKAN_Convolutional_Network = KKAN_Convolutional_Network()    
print(f"============== Training model {model_KKAN_Convolutional_Network.name} ============")
train_model_generic(model_KKAN_Convolutional_Network, mnist_train, mnist_test,device,epochs = 10,path=path)

model_Convs_and_KAN= NormalConvsKAN_Medium() 
print(f"============== Training model {model_Convs_and_KAN.name} ============")
train_model_generic(model_Convs_and_KAN, mnist_train, mnist_test,device,epochs = 10,path=path)

model_SimpleCNN = MediumCNN()
print(f"============== Training model {model_SimpleCNN.name} ============")
train_model_generic(model_SimpleCNN, mnist_train, mnist_test,device,epochs = 10,path=path)

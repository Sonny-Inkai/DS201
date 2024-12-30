import sys
sys.path.insert(1,'Convolutional-KANs')

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import *
from architectures_28x28.conv_and_kan import *

from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
import time
#from hiperparam_tuning import *
#from calflops import calculate_flops
def calculate_time(model,train_obj,test_obj,batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_obj,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_obj,
        batch_size=batch_size,
        shuffle=True)
    start = time.perf_counter()

    train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=1, scheduler=scheduler, path = None,verbose = False,save_last=False,patience = np.inf)
    total_time = time.perf_counter() - start
    print(model.name,"took:",total_time)
    return total_time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformations for CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform)


dataset_name = "CIFAR10"
path = f"models/{dataset_name}"

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists(path):
    os.mkdir(path)

results_path = os.path.join("results",dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)

batch_size = 512
models= [KANC_MLP_Medium(), KKAN_Convolutional_Network(), NormalConvsKAN_Medium(), MediumCNN()]

import json
dictionary={}
for m in models:
    t = calculate_time(m,cifar_train,cifar_test,batch_size)
    dictionary[m.name]=t
with open(f"results/{dataset_name}/epoch_times.json", "w") as outfile: 
    json.dump(dictionary, outfile)
#ライブラリのインポート
import argparse
import operator
import os
import time
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.optim import create_optimizer
from timm.utils import AverageMeter, accuracy
from timm.utils.summary import update_summary
from torch.autograd import Variable
from IPython.display import display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv


#Data Transforms
data_transforms = {
    'test': torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


#set parameters
parser = argparse.ArgumentParser(description="Training Config", add_help=False)

parser.add_argument(
    "--opt",
    default="sgd",
    type=str,
    metavar="OPTIMIZER",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
)

args = parser.parse_args(["--input-size", "3", "224", "224"])

EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 2


#Data Dir Path
data_dir_path = ''+'/' #Specify the path of the folder where the images are located


NUM_FINETUNE_CLASSES = 2 #Specify the number to classify
model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model#.cuda()


#Datasets
image_datasets = {
    'test': torchvision.datasets.ImageFolder(root=data_dir_path,transform=data_transforms['test'])
}


#Data Loader
loader_test = torch.utils.data.DataLoader(image_datasets['test'], batch_size = BATCH_SIZE, shuffle=False)


#Datasets Size
dataset_sizes = {
    'test': len(image_datasets['test'])
}


train_loss_fn = nn.CrossEntropyLoss()#.cuda()
validate_loss_fn = nn.CrossEntropyLoss()#.cuda()

optimizer = create_optimizer(args, model)


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, output_dir=None):
    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    num_updates = epoch * len(loader)
    for _, (input, target) in enumerate(loader):

        data_time_m.update(time.time() - end)

        output = model(input)
        loss = loss_fn(output, target)

        optimizer.zero_grad()

        loss.backward(create_graph=second_order)

        optimizer.step()

        #torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        end = time.time()

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    accuracy_m = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):

            #input = input.cuda()
            #target = target.cuda()

            output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 2))

            reduced_loss = loss.data

            #torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            accuracy_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

    metrics = OrderedDict([("loss", losses_m.avg), ("accuracy", accuracy_m.avg)])

    return metrics


num_epochs = EPOCHS
eval_metric = "accuracy"
best_metric = None
best_epoch = None
compare = operator.gt

#Designation of a folder containing model data after fine tuning
output_dir = ""


model.load_state_dict(
    torch.load(
        os.path.join(output_dir, "best_model.pth")#, map_location=torch.device("cuda")
    )
)


person_positive=0
person_negative=0
p=0
PR = []
LB = []

test_dataloader = loader_test

model.eval()

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        LB = np.concatenate([LB, labels.detach().numpy()], 0)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        PR = np.concatenate([PR, predicted.detach().numpy()], 0)
    while p < len(PR):
        if str(1.0) in str(PR[p].item()):
            person_positive+=1
            p+=1
        else:
            person_negative+=1
            p+=1

if person_positive < person_negative:
    print("LVRR-")
    print("negative=",person_negative)
    print("positive=",person_positive)
else:
    print("LVRR+")
    print("negative=",person_negative)
    print("positive=",person_positive)
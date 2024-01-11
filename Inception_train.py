
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

#Data Transforms
data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]), 
    'val': torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
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



data_dir_path = ''+'/' #Specify the path of the folder where the images are located


NUM_FINETUNE_CLASSES = 2 #Specify the number to classify
model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=NUM_FINETUNE_CLASSES)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model#.cuda()


dataset_trainval = torchvision.datasets.ImageFolder(root=data_dir_path + 'TRAIN',transform=data_transforms['train'])
n_samples = len(dataset_trainval) 
train_size = int(len(dataset_trainval) * 0.8)
val_size = n_samples - train_size

dataset_train, dataset_eval = torch.utils.data.random_split(dataset_trainval, [train_size, val_size])



#Datasets
image_datasets = {
    'train': dataset_train,
    'validation': dataset_eval
}


#Data Loader
loader_train = torch.utils.data.DataLoader(image_datasets['train'], batch_size = BATCH_SIZE, shuffle=True)
loader_eval = torch.utils.data.DataLoader(image_datasets['validation'], batch_size = BATCH_SIZE, shuffle=True)

#Datasets Size
dataset_sizes = {
    'train': len(image_datasets['train']),
    'val': len(image_datasets['validation']),
}


train_loss_fn = nn.CrossEntropyLoss().cuda()
validate_loss_fn = nn.CrossEntropyLoss().cuda()

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
ES = 0
compare = operator.gt

output_dir = "" #Designation of output destination for model data after fine tuning


print("Start")

for epoch in range(0, num_epochs):
    train_metrics = train_one_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn, args, output_dir=output_dir
    )

    eval_metrics = validate(model, loader_eval, validate_loss_fn, args)
    print("epoch")

    if output_dir is not None:
        update_summary(
            epoch,
            train_metrics,
            eval_metrics,
            os.path.join(output_dir, "summary.csv"),
            write_header=best_metric is None,
        )

    metric = eval_metrics[eval_metric]
    if best_metric is None or compare(metric, best_metric):
        best_metric = metric
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        ES = 0
    else:
        ES += 1

    print(epoch)
    print(eval_metrics)
    print("Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
                
    if ES == 5:
        print('Early Stopping!!')
        break


# https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import time
import argparse
from util import *
from convmixer import *


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument('--nowandb', action='store_true', help='disable wandb')

parser.add_argument('--opt', default="adam")
parser.add_argument('--scheduler', default="adam")
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')

parser.add_argument('--dim', default=512, type=int)
parser.add_argument('--heads', default=0, type=int, help='number of different filters in one layer')
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

parser.add_argument('--fix_spatial', action='store_true', help='freeze spatial mixing')
parser.add_argument("--init", type=str, default="random")
parser.add_argument('--input_weight', action='store_true', help='share weights in different layers')
parser.add_argument("--linear_format", action='store_true', help='use linear format conv filters')
parser.add_argument("--no_spatial_bias", action='store_true', help='disable bias for spatial conv')


args = parser.parse_args()
args.spatial = not args.fix_spatial
args.spatial_bias = not args.no_spatial_bias
use_amp = not args.noamp
print(args)


usewandb = not args.nowandb
if usewandb:
    import wandb
    watermark = "{}_h{}".format(args.init,args.heads)
    wandb.init(project=f'convmixer-{args.dataset}',name=watermark)
    wandb.config.update(args)



print(f'==> Preparing {args.dataset} data..')
dataset_mean = (0.4914, 0.4822, 0.4465)
dataset_std = (0.2471, 0.2435, 0.2616)
if args.data_aug:
    train_transform = transforms.Compose([
        transforms.RandAugment(2, 14),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std)
])
if args.dataset == 'cifar10':
    n_class = 10
    image_size = 32
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'cifar100':
    n_class = 100
    image_size = 32
    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvMixer(args.dim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=n_class, image_size=image_size, return_embedding=False,
                  init=args.init, heads=args.heads, spatial=args.spatial, spatial_bias=args.spatial_bias, input_weight=args.input_weight,linear_format=args.linear_format)
if 'cuda' in device:
    print(device)
    print("using data parallel")
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.opt == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

if args.scheduler == "cos":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

num_param = count_parameters(model)
for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    n_batch = 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(X)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
    train_acc = train_acc/n
    train_loss = train_loss/n

        
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)
    test_acc = test_acc/m
    scheduler.step()

    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, "val_acc": test_acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start, 'num_param':num_param})
    else:
        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {test_acc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f},
                 epoch_time: {time.time()-start:.1f}, num_param:{num_param}')
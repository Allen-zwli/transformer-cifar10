import argparse
import os
from threading import Thread

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm    
import torch.utils.data as data
from models import ViT, ResNet, VisT
from torch.autograd import Variable

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms, datasets

from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_poly_lr_decay import PolynomialLRDecay
from torch.utils.tensorboard import SummaryWriter


def train(model_name,patch_size,hidden_size,layers,multihead,heads,lr,n_epochs,n_tokens,name):
    
    writer = SummaryWriter('./runs/'+name)
    device = torch.device('cuda')
    if model_name=='ViT':
        model = ViT.ViT(input_resolution=32, patch_size=patch_size, width=hidden_size, layers=layers, heads=heads, output_dim=10, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif model_name=='ResNet':
        model = ResNet.ResNet18()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = VisT.T_ViT(hidden_size, n_tokens, multihead, heads, 10, 0)
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=4e-5,nesterov=True)
    model = nn.DataParallel(model)
    model = model.to(device)

    # n_epochs = 100
    check_n_batch = 100
    best_loss = 9999

    warm_init_lr = 0
    total_step = 0
    warmup_steps = 400
    lr_step = (lr-warm_init_lr)/warmup_steps
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=40000, end_learning_rate=0, power=1)

    model.train()
    sce_loss = nn.CrossEntropyLoss()
    all_res=[]

    for epoch in range(n_epochs):
        l = 0
        c = 0
        idx = 0
        ra_loss = 0
        ra_sample = 0
        for data_in in train_dl:
            images = Variable(data_in[0]).to(device)
            labels = Variable(data_in[1]).to(device)
            logits = model(images) # n*k
            optimizer.zero_grad()

            batch_loss = sce_loss(logits, labels)

            if total_step <= warmup_steps:
                cur_lr = warm_init_lr + total_step*lr_step
                adjust_learning_rate(optimizer, cur_lr)
            else:
                scheduler.step()

            batch_loss.backward()

            optimizer.step()

            l += batch_loss.item()
            bs = images.shape[0]
            ra_loss += batch_loss.item()*bs
            c += bs
            ra_sample += bs
            if idx%check_n_batch==0:
                print('current {} batch:'.format(check_n_batch), 'running average loss:{:.6f}'.format(ra_loss/ra_sample))
                ra_loss = 0
                ra_sample = 0
            idx += 1
            total_step += 1
            writer.add_scalar('Step_loss/train', batch_loss, total_step)

        #break
        model.eval()
        val_loss, val_acc = eval_class(val_dl, model, device)
        all_res.append([val_loss, val_acc])

        if val_loss < best_loss:
            print('Get best loss on validation set:', val_loss, 'Saving the best model')
            best_loss = val_loss
            torch.save(model.state_dict(), 'weights/'+name+'.pt')
        print("Epoch: {}/{}...".format(epoch, n_epochs),
              "Training Loss: {:.6f}...".format(l/c),
              "Val Loss: {:.6f}".format(val_loss),
              "Val Acc: {:.4f}".format(val_acc))

        writer.add_scalar('Epoch_loss/train', l/c, epoch)
        writer.add_scalar('Epoch_loss/val', val_loss, epoch)
        writer.add_scalar('Epoch_acc/val', val_acc, epoch)

        model.train()
        
    return all_res
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='run.py')
    parser.add_argument('--model', default='ResNet', help='which model to use, ResNet18 (ResNet) or Vision Transformer (ViT) or Visual Transformer (VisT)')
    parser.add_argument('--batch-size', type=int, default=128, help='size of each image batch')
    parser.add_argument('--hidden-size', type=int, default=256, help='hidden size for linear layer')
    parser.add_argument('--patch-size', type=int, default=4, help='patch size')
    parser.add_argument('--layers', type=int, default=6, help='number of layers of transformer blocks')
    parser.add_argument('--multihead', action='store_true', help='use multihead attention')
    parser.add_argument('--heads', type=int, default=6, help='number of heads of multihead attention layer')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--n_tokens', type=int, default=4, help='patch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--name', default='ResNet18', help='save to weights/name.pt')
    opt = parser.parse_args()
    print(opt)
    
    train_transform = transforms.Compose([ 
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            ]) 
    test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    cifar10_train = datasets.CIFAR10('./',train=True, transform=train_transform, download=True)
    cifar10_test = datasets.CIFAR10('./', train=False, transform=test_transform, download=True)
    batch_size = 128
    train_dl = data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    val_dl = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True)
    
    train(opt.model,opt.patch_size,opt.hidden_size,opt.layers,opt.multihead,opt.heads,opt.lr,opt.epochs,opt.n_tokens,opt.name)
    
    
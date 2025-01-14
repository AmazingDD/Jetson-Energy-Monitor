import os
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *
from data import load_data
import power_check as pc

model_conf = {
    'vgg': VGG9,
    'svgg': SVGG9, 
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='energy checker with jetson series edge devices')
    parser.add_argument('-T', type=int, default=4, help='time step (default: 4)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-epochs', type=int, default=2, help='training epochs')
    parser.add_argument('-model', type=str, default='vgg', help='neural network used in training')
    parser.add_argument('-dataset', type=str, default='cifar10', help='dataset name for measure')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    train_set, test_set, C, H, W, num_classes = load_data(args.dataset)

    # training can be implemented on server GPU 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    model = model_conf[args.model](num_classes, C, H, W, T=args.T)
    model.to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    pc.printFullReport(pc.getDevice())
    pl = pc.PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')

    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1:2d}/{args.epochs}]')

        model.train()
        for images, targets in tqdm(train_loader, unit='batch', ncols=100, desc='Train'):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimzer.step()

    time.sleep(5)
    pl.stop()
    filename = f'./{args.model}/{args.dataset}/train/'
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    pc.printFullReport(pc.getDevice())

    torch.save(model.state_dict(), f'params_{args.model}_{args.dataset}.pt')

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    model = model_conf[args.model](num_classes, C, H, W, T=args.T)
    model.load_state_dict(torch.load(f'params_{args.model}_{args.dataset}.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    pc.printFullReport(pc.getDevice())
    pl = pc.PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')
    
    for images, _ in tqdm(test_loader, unit='batch', ncols=100, desc='Inference'):
        model(images.to(device))

    time.sleep(5)
    pl.stop()
    filename = f'./{args.model}/{args.dataset}/test/'
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    pc.printFullReport(pc.getDevice())

    
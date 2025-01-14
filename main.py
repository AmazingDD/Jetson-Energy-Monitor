import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *
from data import load_data
from power_check import *

model_conf = {
    'vgg': VGG9,
    'svgg': SVGG9, 
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='energy checker with jetson series edge devices')
    parser.add_argument('-T', type=int, default=4, help='time step (default: 4)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-epochs', type=int, default=70, help='training epochs')
    parser.add_argument('-model', type=str, default='vgg9', help='neural network used in training')
    parser.add_argument('-dataset', type=str, default='cifar10', help='dataset name for measure')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set, test_set, C, H, W, num_classes = load_data(args.dataset)

    # training can be implemented on server GPU 
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)å
    model = model_conf[args.model](num_classes, C, H, W, T=args.T)
    model.to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(f'./{args.model}/{args.dataset}/train/'):
        os.makedirs(f'./{args.model}/{args.dataset}/train/')

    pc.printFullReport(pc.getDevice())
    pl = pc.PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')

    train_acc, train_loss, train_samples = 0, 0., 0
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1:2d}/{args.epochs}]')

        model.train()
        for images, targets in tqdm(train_loader, unit='batch', ncols=100):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimzer.step()

            train_samples += targets.numel()
            train_loss += loss.item() * targets.numel()
            train_acc += (outputs.argmax(1) == targets).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples

        print(f'train loss: {train_loss:.4f} train acc.: {train_acc:.2f}')

    time.sleep(5)
    pl.stop()
    filename = f'./{args.model}/{args.dataset}/train/'
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    pc.printFullReport(pc.getDevice())

    torch.save(model.state_dict(), f'params_{args.model}_{args.dataset}.pt')

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    iter_loader = iter(test_loader)

    model = model_conf[args.model](num_classes, C, H, W, T=args.T)
    model.load_state_dict(torch.load(f'params_{args.model}_{args.dataset}.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    if not os.path.exists(f'./{args.model}/{args.dataset}/test/'):
        os.makedirs(f'./{args.model}/{args.dataset}/test/')

    pc.printFullReport(pc.getDevice())
    pl = pc.PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    pl.recordEvent(name='Process Start')
    
    for _ in range(200):
        images, _ = next(iter_loader)
        model(images)

    time.sleep(5)
    pl.stop()
    filename = f'./{args.model}/{args.dataset}/test/'
    pl.showDataTraces(filename=filename)
    print(str(pl.eventLog))
    pc.printFullReport(pc.getDevice())

    
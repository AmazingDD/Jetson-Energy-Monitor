import os
import pickle
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, root='./tiny-imagenet-200', train=True, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.train = train

        if self.train:
            with open(os.path.join(self.root, 'tiny-imagenet-200' , 'train_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)
        else:
            with open(os.path.join(self.root, 'tiny-imagenet-200', 'val_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)

        self.targets = self.targets.type(torch.LongTensor)

    def __getitem__(self, index):
        data = self.data[index].permute(1, 2, 0).numpy()
        data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)

        return data, self.targets[index] 
    
    def __len__(self):
        return len(self.targets)

def load_data(name='cifar10'):
    if name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
            transforms.RandomHorizontalFlip(),     # 随机水平翻转
            transforms.ToTensor(),                 # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 测试集无需数据增强
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=train_transform
        )

        test_set = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=test_transform
        )

        C, H, W, num_classes = 3, 32, 32, 10

    elif name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        train_set = TinyImageNetDataset('./data', train=True, transform=transform_train)
        test_set = TinyImageNetDataset('./data', train=False, transform=transform_test)

        C, H, W, num_classes = 3, 64, 64, 200
    else:
        raise ValueError(f'Invalid dataset name: {name}')
    
    
    return train_set, test_set, C, H, W, num_classes

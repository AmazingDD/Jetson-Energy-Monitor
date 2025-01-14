import torchvision
import torchvision.transforms as transforms

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
    else:
        raise ValueError(f'Invalid dataset name: {name}')
    
    return train_set, test_set, C, H, W, num_classes


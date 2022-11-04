import os
import sys
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from settings import PROJECT_ROOT, DATA_DIR

__all__ = ['mnist', 'cifar10']


class MNIST(datasets.MNIST):
    def __init__(self, root, train):
        super().__init__(root, train=train, download=False)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train):
        super().__init__(root, train=train, download=False)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) 


def mnist(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    config = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    return {
        'train': DataLoader(MNIST(root, train=True), **config),
        'val': DataLoader(MNIST(root, train=False), **config)
    }

def cifar10(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    config = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    return {
        'train': DataLoader(CIFAR10(root, train=True), **config),
        'val': DataLoader(CIFAR10(root, train=False), **config)
    }
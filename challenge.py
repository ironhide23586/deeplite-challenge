import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from core.model.lenet import LeNet
from torch.utils import data

"""
    Deeplite coding challenge.
    The main goal of this coding challenge is to implement a very simple pruning algorithm for lenet. There are two 
    steps to implement this coding challenge. 
    Step 1:
        Implement the pruning algorithm to remove weights which are smaller than the given threshold (prune_model)
    Step 2:
        As you may know after pruning, the accuracy drops a lot. To recover the accuracy, we need to do the fine-tuning.
        It means, we need to retrain the network for few epochs. Use prune_model method which you have implemented in 
        step 1 and then fine-tune the network to regain the accuracy drop (prune_model_finetune)

    *** The pretrained lenet has been provided (lenet.pth)
    *** You need to install torch 0.3.1 on ubuntu
    *** You can use GPU or CPU for fine-tuning
"""


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def transform_train_op():
    t = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.Scale(32),
                            transforms.ToTensor(),
                            transforms.Normalize([.1307], [.3081])])
    return t


def transform_test_op():
    t = transforms.Compose([transforms.Scale(32),
                            transforms.ToTensor(),
                            transforms.Normalize([.1307], [.3081])])
    return t


def get_mnist_dataloaders(root, batch_size):
    """
    This function should return a pair of torch.utils.data.DataLoader.
    The first element is the training loader, the second is the test loader.
    Those loaders should have been created from the MNIST dataset available in torchvision.

    For the training set, please preprocess the images in this way:
        - Resize to 32x32
        - Randomly do a horizontal flip
        - Normalize using mean 0.1307 and std 0.3081

    For the training set, please preprocess the images in this way:
        - Resize to 32x32
        - Normalize using mean 0.1307 and std 0.3081

    :param root: Folder where the dataset will be downloaded into.
    :param batch_size: Number of samples in each mini-batch
    :return: tuple
    """
    force_makedir(root)
    mnist_data_train = datasets.MNIST(root, train=True, download=True, transform=transform_train_op())
    mnist_data_test = datasets.MNIST(root, train=False, download=True, transform=transform_test_op())
    mnist_train_data_loader = data.DataLoader(mnist_data_train, batch_size=batch_size, shuffle=True,
                                              num_workers=cpu_count())
    mnist_test_data_loader = data.DataLoader(mnist_data_test, batch_size=1, shuffle=False, num_workers=cpu_count())
    return mnist_train_data_loader, mnist_test_data_loader


def get_accuracy_top1(model, data_loader):
    """
    This function should return the top1 accuracy% of the model on the given data loader.
    :param model: LeNet object
    :param data_loader: torch.utils.data.DataLoader
    :return: float
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        k = 0


def prune_model(model, threshold):
    """
    This function should set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    :param model: LeNet object
    :param threshold: float
    """
    pass


def prune_model_finetune(model, train_loader, test_loader, threshold):
    """
    This function should first set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    Then, it should finetune the model by making sure that the weights that were set to zero remain zero after the gradient descent steps.
    :param model: LeNet object
    :param train_loader: training set torch.utils.data.DataLoader
    :param test_loader: testing set torch.utils.data.DataLoader
    :param threshold: float
    """
    pass


if __name__ == '__main__':
    model = torch.load(open('lenet.pth', 'rb'))
    # Do your things

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, test_loader = get_mnist_dataloaders('data', 16)
    acc = get_accuracy_top1(model, test_loader)


import os
from multiprocessing import cpu_count
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from core.model.lenet import LeNet
from torch.utils import data

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
    mnist_test_data_loader = data.DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=cpu_count())
    return mnist_train_data_loader, mnist_test_data_loader


def get_accuracy_top1(model, data_loader):
    """
    This function should return the top1 accuracy% of the model on the given data loader.
    :param model: LeNet object
    :param data_loader: torch.utils.data.DataLoader
    :return: float
    """
    preds = []
    truths = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for data, target in data_loader:
        data = data.to(device)
        output = model.infer_out(data)
        preds.append(output.argmax(1).cpu().numpy())
        truths.append(target.numpy())
    preds = np.hstack(preds)
    truths = np.hstack(truths)
    acc = accuracy_score(truths, preds)
    return acc


def prune_model(model, threshold):
    """
    This function should set the model's weight to 0 if their absolutes values are lower or equal to the given threshold.
    :param model: LeNet object
    :param threshold: float
    """
    num_params_pruned = 0
    total_num_params = 0
    num_zero_params = 0
    for name, param in model.state_dict().items():
        var = param.detach().cpu().numpy()
        total_num_params += var.flatten().shape[0]
        filt = np.logical_and(np.abs(var) <= threshold, np.abs(var) > 0.)
        params_pruned = var[filt].flatten().shape[0]
        var[filt] = 0.
        num_zero_params += var[var == 0.].flatten().shape[0]
        param.copy_(torch.cuda.FloatTensor(var))
        num_params_pruned += params_pruned
    pruned_fraction = num_params_pruned / total_num_params
    zero_fraction = num_zero_params / total_num_params
    return num_params_pruned, total_num_params, pruned_fraction, zero_fraction


def prune_model_finetune(model, train_loader, test_loader, threshold, out_dir, train_epochs=2, learn_rate=1e-4,
                         eval_freq=20):
    """
    This function should first set the model's weight to 0 if their absolutes values are lower or equal to the given
    threshold.
    Then, it should finetune the model by making sure that the weights that were set to zero remain zero after the
    gradient descent steps.
    :param model: LeNet object
    :param train_loader: training set torch.utils.data.DataLoader
    :param test_loader: testing set torch.utils.data.DataLoader
    :param threshold: float
    """
    model_out_dir = out_dir + os.sep + 'trained_models'
    force_makedir(model_out_dir)
    log_fpath = out_dir + os.sep + 'log.txt'
    num_params_pruned, total_num_params, pruned_fraction, zero_fracion = prune_model(model, threshold)
    print_and_log('Number of pruned params = ' + str(num_params_pruned) + ' out of ' + str(total_num_params) +
                  ' total params', log_fpath)
    print_and_log('Fraction of params pruned = ' + str(100. * pruned_fraction) + '%', log_fpath)
    print_and_log('Fraction of zero params = ' + str(100. * zero_fracion) + '%', log_fpath)

    acc = get_accuracy_top1(model, test_loader)
    print_and_log('Accuracy after pruning = ' + str(acc), log_fpath)

    loss_plot_save_path = out_dir + os.sep + 'loss_plot.png'
    accuracy_plot_save_path = out_dir + os.sep + 'accuracy_plot.png'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_op = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    loss_plot = []
    acc_plot = [[0, acc]]
    train_step = 0
    model_fname = '_'.join(['lenet_pruned', str(train_step) + 'step', str(acc) + 'accuracy']) + '.torchmodel'
    model_save_path = model_out_dir + os.sep + model_fname
    torch.save(model.state_dict(), model_save_path)

    max_acc = acc
    best_model_path = out_dir + os.sep + model_fname
    torch.save(model.state_dict(), best_model_path)

    for epoch in range(train_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model.forward(x)
            loss = loss_op(outputs, y)
            loss_plot.append([train_step, loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_and_log('Iter ' + str(train_step) + ', loss = ' + str(loss.item()), log_fpath)
            loss_plot.append([train_step, loss.item()])
            prune_model(model, threshold)
            train_step += 1

            if train_step % eval_freq == 0:
                acc = get_accuracy_top1(model, test_loader)
                acc_plot.append([train_step, acc])
                model_fname = '_'.join(['lenet_pruned', str(train_step) + 'step', str(acc) + 'accuracy']) \
                              + '.torchmodel'
                model_save_path = model_out_dir + os.sep + model_fname
                torch.save(model.state_dict(), model_save_path)
                if acc > max_acc:
                    max_acc = acc
                    os.remove(best_model_path)
                    best_model_path = out_dir + os.sep + model_fname
                    torch.save(model.state_dict(), best_model_path)

                losses = np.array(loss_plot)
                accs = np.array(acc_plot)

                plt.clf()
                plt.plot(losses[:, 0], losses[:, 1])
                plt.xlabel('Training Iterations')
                plt.ylabel('Cross Entropy Loss')
                plt.savefig(loss_plot_save_path)

                plt.clf()
                plt.plot(accs[:, 0], accs[:, 1])
                plt.xlabel('Training Iterations')
                plt.ylabel('Top-1 Validation Accuracies')
                plt.savefig(accuracy_plot_save_path)
    print_and_log('Done!', log_fpath)


def get_threshold(model):
    params = [p.detach().cpu().numpy() for p in model.parameters()]
    means = [p.mean() for p in params]
    stds = [p.std() for p in params]
    params_mean = np.mean(means)
    params_std = np.mean(stds)
    thres = abs(abs(params_mean) - 3.0 * abs(params_std))
    return thres


def print_and_log(line, log_fpath):
    print(line)
    with open(log_fpath, 'a+') as f:
        f.write(line + '\n')


if __name__ == '__main__':
    train_write_dir = 'train_sessions'
    out_dir = train_write_dir + os.sep + 'train_sess_0'
    if not os.path.isdir(out_dir):
        load_model_fpath = 'lenet.torchmodel'
        if not os.path.isfile(load_model_fpath):
            model_tmp = torch.load(open('lenet.pth', 'rb'))
            torch.save(model_tmp.state_dict(), load_model_fpath)
    else:
        training_dirs = glob(train_write_dir + os.sep + '*')
        train_sess_indices = [int(dir.split('_')[-1]) for dir in training_dirs]
        idx = np.argmax(train_sess_indices)
        selected_train_sess_idx = train_sess_indices[idx]
        out_dir = training_dirs[idx].replace(str(idx), str(idx + 1))
        src_model_dir = training_dirs[idx]
        model_fpaths = glob(src_model_dir + os.sep + 'trained_models' + os.sep + '*')
        src_accs = [float(p.split('_')[-1].split('accuracy')[0]) for p in model_fpaths]
        load_model_fpath = model_fpaths[np.argmax(src_accs)]

    force_makedir(out_dir)
    log_fpath = out_dir + os.sep + 'log.txt'

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    print_and_log('Loading model from ' + load_model_fpath, log_fpath)
    model.load_state_dict(torch.load(load_model_fpath))

    train_loader, test_loader = get_mnist_dataloaders('data', 64)

    acc = get_accuracy_top1(model, test_loader)
    print_and_log('Initial accuracy = ' + str(acc), log_fpath)

    threshold = get_threshold(model)
    print_and_log('Pruning Threshold = ' + str(threshold), log_fpath)
    prune_model_finetune(model, train_loader, test_loader, threshold, out_dir)

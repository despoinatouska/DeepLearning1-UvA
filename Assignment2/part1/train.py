################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

import os
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from cifar100_utils import get_train_validation_set, get_test_set, set_dataset


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Randomly initialize and modify the model's last layer for CIFAR100.
    for param in model.parameters(): # Freeze all layers except the last one
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.fc.weight.data.normal_(std=0.01)
    model.fc.bias.data.fill_(0)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set  = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=augmentation_name)
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = data.DataLoader(val_set, batch_size, shuffle=False, drop_last=False, num_workers=2)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    model_name = "resnet18_Cifar100"
    loss_module = nn.CrossEntropyLoss()
    val_scores = []
    train_losses, train_scores = [], []
    best_val_epoch = -1
    for epoch in range(epochs):
        ############
        # Training #
        ############
        model.train()
        true_preds, count = 0., 0
        # t = tqdm(train_loader, leave=False)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad() # zero out the gradients of all the parameters that the optimizer will update
            preds = model(imgs)
            loss = loss_module(preds, labels)
            loss.backward() # Backpropagation - calculate gradients
            optimizer.step() # Update parameters
            # Record statistics during training
            true_preds += (preds.argmax(dim=-1) == labels).sum().item() # 128x100
            count += labels.shape[0]
            train_losses.append(loss.item())
        train_acc = true_preds / count
        train_scores.append(train_acc)

        ##############
        # Validation #
        ##############
        val_acc = evaluate_model(model, val_loader, device)
        val_scores.append(val_acc)
        print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%")

        if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
            print("\t   (New best performance, saving model...)")
            save_model(model, checkpoint_name, model_name)
            best_val_epoch = epoch

    # Load the best model on val accuracy and return it.
    model = load_model(checkpoint_name, model_name, net=model)
    test_acc = evaluate_model(model, val_loader, device)

    # Plot a curve of the validation accuracy
    plt.plot([i for i in range(1,len(train_scores)+1)], train_scores, label="Train")
    plt.plot([i for i in range(1,len(val_scores)+1)], val_scores, label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.ylim(min(val_scores), max(train_scores)*1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(checkpoint_name,'validation_performance.png'))

    #######################
    # END OF YOUR CODE    #
    #######################

    return model

def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")

def save_model(model, model_path, model_name):
    os.makedirs(model_path, exist_ok=True)
    model_file =_get_model_file(model_path, model_name)
    torch.save(model.state_dict(), model_file)

def load_model(model_path, model_name, net=None):
    model_file = _get_model_file(model_path, model_name)
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    net.load_state_dict(torch.load(model_file))
    return net

def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    true_preds, count = 0., 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs).argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
    accuracy = true_preds / count

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise, checkpoint_dir):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = get_model().to(device)

    # Get the augmentation to use
    augmentation_name = augmentation_name
    print(augmentation_name)

    # Train the model
    checkpoint_name = checkpoint_dir #"saved_models/"
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=augmentation_name)

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir, test_noise)
    val_loader = data.DataLoader(test_set, batch_size, shuffle=False, drop_last=False, num_workers=2)
    test_acc = evaluate_model(model, val_loader, device)
    print((f" Test accuracy: {test_acc*100.0:4.2f}% ").center(50, "=")+"\n")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cifar10'],
                        help='Dataset to use.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')
    parser.add_argument('--checkpoint_dir', default='saved_models/', type=str,
                        help='Checkpoint directory')

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop('dataset'))
    main(**kwargs)

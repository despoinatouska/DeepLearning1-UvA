  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    num_class = predictions.shape[1]
    predictions = predictions.argmax(axis=1)
    conf_mat = np.zeros((num_class, num_class))
    for actual, pred in zip(predictions, targets):
      conf_mat[pred][actual] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    num_classes = confusion_matrix.shape[0]
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_beta = np.zeros(num_classes)
    for i in range(num_classes):
      tp = confusion_matrix[i,i]
      fp = np.sum(confusion_matrix[i,:]) - tp
      fn = np.sum(confusion_matrix[:,i]) - tp
      precision[i] = tp / (tp + fp) if tp + fp > 0 else 0
      recall[i] = tp / (tp + fn) if tp + fn > 0 else 0
      f1_beta[i] = ((1 + beta**2) * (precision[i] * recall[i])) / (beta**2 * precision[i]  +recall[i]) if (beta**2 * precision[i]  +recall[i]) > 0 else 0
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_beta": f1_beta}
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics

def plot_cm(conf_mat):
    true_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_mat, cmap='viridis')
    plt.colorbar()
    num_classes = len(true_labels)
    plt.xticks(np.arange(num_classes), true_labels, rotation=45)
    plt.yticks(np.arange(num_classes), true_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(i, j, conf_mat[i, j], ha='center', va='center', color='white')
    plt.tight_layout()
    plt.savefig("confusion_matrix_pytorch.png")
    plt.show()

def sort_lists(list_to_be_sort, labels, name):
    sorted_indices = sorted(range(len(list_to_be_sort)), key=lambda k: list_to_be_sort[k])
    sorted_tl = [labels[i] for i in sorted_indices]
    print(sorted_tl)
    print(sorted(list_to_be_sort))
    print()
    plot_list(sorted(list_to_be_sort), sorted_tl, name)
    # plot_table(sorted(list_to_be_sort), [name], sorted_tl, name)

def plot_list(values, labels, name):
    formatted_values = [f'{value:.3f}' for value in values]
    table_data = [[labels[i], formatted_values[i]] for i in range(min(len(labels), len(formatted_values)))]
    # plt.figure(figsize=(8, 4))
    tab = plt.table(cellText=table_data,
                    colLabels=['Labels', name],
                    loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    # tab.scale(1.5, 1.5)
    plt.axis('off')
    plt.title('Table Plot')
    plt.show()
    # plot_table(sorted(list_to_be_sort), [name], sorted_tl, name)

def plot_table(table, row_labels, column_labels, name):
        table = [[f'{value:.3f}' for value in row] for row in table]
        plt.figure(figsize=(8, 4))
        table = plt.table(cellText=table,
                          rowLabels=row_labels,
                          colLabels=column_labels,
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.axis('off')  # Hide the axes
        plt.title(name)
        plt.show()
def calc_f1(beta_list, precision, recall, labels):
    print()
    print(labels)
    num_classes = len(labels)
    num_beta = len(beta_list)
    f1_beta = np.zeros((num_beta, num_classes))
    for j, beta in enumerate(beta_list):
        for i in range(num_classes):
            f1_beta[j,i] = ((1 + beta**2) * (precision[i] * recall[i])) / (beta**2 * precision[i]  +recall[i]) if (beta**2 * precision[i]  +recall[i]) > 0 else 0
        print("F1 score wth beta {}".format(beta))
        print(f1_beta[j,:])

    plot_table(f1_beta, beta_list, labels, "F1 score for each class")


def evaluate_model(model, data_loader, num_classes=10, mode="test"):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    loss_module = nn.CrossEntropyLoss()
    epoch_loss = 0
    num_batches = 0
    conf_mat = np.zeros((num_classes, num_classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    true_preds, count = 0., 0
    # for imgs, labels in tqdm(data_loader, leave=False):
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs)
            true_preds += (preds.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
            loss = loss_module(preds, labels)
            epoch_loss += loss.item()
            num_batches += 1
            conf_mat += confusion_matrix(preds, labels)
    test_acc = true_preds / count
    metrics = confusion_matrix_to_metrics(conf_mat)
    # print(test_acc, metrics["accuracy"])
    val_losses = epoch_loss / num_batches
    if mode == "test":
        plot_cm(conf_mat)
        beta = [0.1, 1, 10]
        true_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # Precision
        print("Precision for each class (sorted):")
        sort_lists(metrics["precision"], true_labels, "Precision")
        # Recall
        print("Recall for each class (sorted):")
        sort_lists(metrics["recall"], true_labels, "Recall")
        # F1 scores
        calc_f1(beta, metrics["precision"], metrics["recall"], true_labels)
    #######################
    # END OF YOUR CODE    #
    #######################
    return val_losses, metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    training_dataloader = cifar10_loader['train']
    validation_dataloader = cifar10_loader['validation']
    testing_loader = cifar10_loader['test']
    # TODO: Initialize model and loss module
    n_classes = 10 # number of classes in the CIFAR10 dataset
    n_inputs = 32*32*3 # size of images in CIFAR10 dataset
    model = MLP(n_inputs, hidden_dims, n_classes, use_batch_norm)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr) # Default parameters, feel free to change
    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []
    best_valid_acc = 0.0
    best_model = []
    for epoch in range(epochs):
      ############
      # Training #
      ############
      model.train()
      true_preds, count = 0., 0
      epoch_loss = 0
      num_batches = 0
      # for imgs, labels in tqdm(training_dataloader, desc=f"Epoch {epoch+1}", leave=False):
      for imgs, labels in training_dataloader:
          imgs, labels = imgs.to(device), labels.to(device) # To GPU
          optimizer.zero_grad()
          preds = model(imgs)
          loss = loss_module(preds, labels)
          # print(loss)
          epoch_loss += loss.item()
          num_batches += 1
          # print(epoch_loss)
          loss.backward()
          optimizer.step()
          # Record statistics during training
          true_preds += (preds.argmax(dim=-1) == labels).sum()
          count += labels.shape[0]
      train_acc = true_preds / count
      train_accuracies.append(train_acc.item())
      train_losses.append(epoch_loss / num_batches) # store the average score of each epoch
      ##############
      # Validation #
      ##############
      val_loss, metrics = evaluate_model(model, validation_dataloader, num_classes=10, mode="val")
      val_accuracies.append(metrics["accuracy"])
      val_losses.append(val_loss)
      print("Epoch {}/{}: Train. Loss: {}, Train. Acc.: {}, Val. Loss: {}, Val. Acc.: {}.".\
            format(epoch+1, epochs, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1]))

      # Save the best model ...
      if best_valid_acc < val_accuracies[-1]:
        print("Save parameter in epoch {} .".format(epoch + 1))
        best_valid_acc = val_accuracies[-1]
        best_model = deepcopy(model)


    # TODO: Test best model
    test_loss, test_metrics = evaluate_model(best_model, testing_loader, num_classes=10)
    test_accuracy = test_metrics["accuracy"]
    print("Test Acc.: {}".format(test_accuracy))
    # TODO: Add any information you might want to save for plotting
    logging_info = dict({"train_loss": train_losses, "train_ac": train_accuracies, "valid_loss": val_losses, "valid_ac": val_accuracies})
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info

def plot_logging_info(list_train, list_valid, name):
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.plot(list_train, label="Training Set")
    plt.plot(list_valid, label="Validation Set")
    plt.legend()
    plt.savefig(name + "_pytorch" + ".png")
    plt.show()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    plot_logging_info(logging_info['train_loss'], logging_info['valid_loss'], 'Loss')
    plot_logging_info(logging_info['train_ac'], logging_info['valid_ac'], 'Accuracy')
    
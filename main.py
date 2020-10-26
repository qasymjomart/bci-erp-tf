#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 2020

@author: Kassymzhomart Kunanbayev aka @qasymjomart

credits: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.benchmark = True
# import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from train_tools import train_model_without_validation, initialize_model
from dataset_import import SI_P300Datasets, Reshape
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Models to choose from [resnet18, resnet50, resnet101, alexnet, vgg11_bn, vgg16, vgg19, squeezenet, densenet121, densenet161, densenet201, inception, googlenet]
model_names = ['resnet18', 'resnet50', 'resnet101', 'alexnet', 'vgg11_bn', 'vgg16', 'vgg19', 'squeezenet', 'densenet121', 'densenet161', 'densenet201', 'inception', 'googlenet']
# model_name = "resnet101"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

#%%

dataset_name = 'TenHealthyData'
df = pd.DataFrame()
accuracies = {}

for model_name in model_names:
    subjects = [0,1,2,3,4,5,6,7,8,9]
    accuracies[model_name] = []
    for sub in subjects:
        sub_array = []
        sub_array = subjects
        sub_array.remove(sub)
    
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # Print the model we just instantiated
        print(model_ft)
        
        #%nn%
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = SI_P300Datasets(dataset_name,
        								transform = transforms.Compose([
							            	Reshape(input_size),
							            	normalize,
							        ])) 
        train_dataset.import_subjects(sub_array)
        train_dataset.apply_normalization()
        scaler = train_dataset.get_normalization_params()
        
        test_dataset = SI_P300Datasets(dataset_name, 
        								transform = transforms.Compose([
        									Reshape(input_size),
        									normalize,
        								]))
        test_dataset.import_subjects([sub])
        test_dataset.apply_normalization_to_test(scaler)
        
        train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        # Send the model to GPU
        model_ft = model_ft.to(device)
        
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.0001)
        
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        
        # Train and evaluate
        model_ft = train_model_without_validation(model_ft, model_name, train_loader, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))
        
        
        # Testing
        cuda = 1 if torch.cuda.is_available() else 0
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor   
        print('-'*20)
        print('Finished Training')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.type(Tensor).to(device), labels.type(dtype = torch.long).to(device)
                outputs = model_ft(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(accuracy)
        accuracies[model_name].append(accuracy)
        
        del train_dataset, train_loader, test_dataset, test_loader, model_ft, params_to_update, criterion, optimizer_ft, device
    
    df[model_name] = accuracies[model_name]
    accuracies
    with open(model_name + '_' + dataset_name +  '_accuracies.txt', 'w') as file:
        file.write(str(accuracies[model_name]))

df.to_csv(dataset_name + '_accuracies.csv')
    

#%%
# Initialize the non-pretrained version of the model used for this run
# scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
# scratch_model = scratch_model.to(device)
# scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
# scratch_criterion = nn.CrossEntropyLoss()
# _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# # Plot the training curves of validation accuracy vs. number
# #  of training epochs for the transfer learning method and
# #  the model trained from scratch
# ohist = []
# shist = []

# ohist = [h.cpu().numpy() for h in hist]
# shist = [h.cpu().numpy() for h in scratch_hist]

# plt.title("Validation Accuracy vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
# plt.plot(range(1,num_epochs+1),shist,label="Scratch")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1, num_epochs+1, 1.0))
# plt.legend()
# plt.show()
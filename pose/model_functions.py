import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from collections import OrderedDict


# Train the classifier
def train_classifier(model, optimizer, criterion, arg_epochs, train_loader, validate_loader, gpu):


    epochs = arg_epochs
    steps = 0
    print_every = 40

    model.to(gpu)

    for e in range(epochs):
    
        model.train()

        running_loss = 0

        for el in iter(train_loader):

            images = el['image']
            labels = el['class']
    
            steps += 1
    
            images, labels = images.to(gpu), labels.to(gpu)
    
            optimizer.zero_grad()
    
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            if steps % print_every == 0:
            
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validate_loader, criterion, gpu)
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
        
                running_loss = 0
                model.train()



# Function for measuring network accuracy on test data
def test_accuracy(model, test_loader, gpu):

    # Do validation on the test set
    model.eval()
    model.to(gpu)

    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in iter(test_loader):
    
            images, labels = images.to(gpu), labels.to(gpu)
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    

# Function for saving the model checkpoint
def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 102,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'checkpoint.pth')
        

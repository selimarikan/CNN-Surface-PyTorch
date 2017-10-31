import torch
import torchvision
from torchvision import transforms
import numpy as np
import gc
import time
import math
import copy
import os
import argparse
import cnnUtils
import neuralNets

# Steps
# ----------------------------------
# - Choose network arch
# - Load weights
# - 
# - Output class probabilities. Write code suitable for binary and multi-class
# - Write all results to a new TXT file


float_formatter = lambda x: "%.2f" % x

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind':float_formatter})

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='dsmnl', type=str, help='choose network architecture to be used. Options are: dsmnl, alexnet, resnet, vgg')
    parser.add_argument('--checkpoint', required=True, help='path of neural network to use')
    parser.add_argument('--images', required=True, help='path of images to predict')
    parser.add_argument('--batchSize', type=int, default=10, help='how many images to be loaded in one batch')
    parser.add_argument('--imageSize', type=int, default=128, help='what should be the image size for network input')

    opt = parser.parse_args()
    
    # Should be average in the end.
    setMean = [0.469, 0.469, 0.469]
    setStd = [0.049, 0.049, 0.049]

    # Create net
    net, criterion = neuralNets.CreateNet(opt.net, setMean, setStd, opt.imageSize, outputClassCount=2)
    
    # 1. Load the network
    if opt.checkpoint == '':
        print('Network path cannot be empty')

    checkpoint = torch.load(opt.checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(opt.checkpoint, checkpoint['epoch']))

    print(net)
    if net is not None:
        print('Loaded the network successfully')
    if torch.cuda.is_available():
        net = net.cuda()

    # Set model for testing
    net.eval()
    
    # 2. Load the images
    if opt.images == '':
        print('Image path cannot be empty')

    setImageSize = opt.imageSize

    # 2.1. Set the image transforms
    dataTransform = {
    'test': transforms.Compose([
        transforms.Scale(setImageSize),
        transforms.CenterCrop(setImageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=setMean, std=setStd)
    ]), }

    # 2.2. Create dataset and loader
    dataset = torchvision.datasets.ImageFolder(opt.images, dataTransform['test'])
    #datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(dataset, batch_size=950, shuffle=False, num_workers=1, pin_memory=True)
    #print(datasetLoader)
    datasetSize = len(dataset)
    datasetClasses = dataset.classes

    print(str(datasetSize) + ' images will be used.' )
    print('GPU will ' + ('' if torch.cuda.is_available() else 'not ') + 'be used.' )
    print(str(len(datasetClasses)) + ' output classes')

    # Evaluate the network
    confusionMat = cnnUtils.CalculateConfusion(net, datasetClasses, testLoader)
    infTime = cnnUtils.EvaluateInference(net, testLoader)
    infText = 'Inference time: ' + infTime
    print(infText)

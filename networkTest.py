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
from neuralNets import DSMNLNet

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
    net = None
    
    # Should be average in the end.
    setMean = [0.5, 0.5, 0.5]
    setStd = [0.5, 0.5, 0.5]

    # 1. Load the network
    if opt.checkpoint == '':
        print('Network path cannot be empty')

    net = torch.load(opt.network)
    if net is not None:
        print('Loaded the network successfully')
    if torch.cuda.is_available():
        net = net.cuda()

    # 2. Load the images
    if opt.images == '':
        print('Image path cannot be empty')

    setImageSize = opt.imageSize

    # 2.1. Set the image transforms
    dataTransform = {
    'test': transforms.Compose([
        transforms.Scale(setImageSize),
        transforms.RandomCrop(setImageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=setMean, std=setStd)
    ]), }

    # 2.2. Create dataset and loader
    dataset = torchvision.datasets.ImageFolder(opt.images, dataTransform['test'])
    datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(datasetLoader)
    datasetSize = len(dataset)
    datasetClasses = dataset.classes

    print(str(datasetSize) + ' images will be used.' )
    print('GPU will ' + ('' if torch.cuda.is_available() else 'not ') + 'be used.' )
    print(str(len(datasetClasses)) + ' output classes')

    # 2.3. Load a sample batch
    sampleInputs, sampleClasses = next(iter(datasetLoader))

    # 2.4. Sanity check
    if sampleInputs.size()[0] != sampleClasses.size()[0]:
        print('Dataset is not loaded correctly. ')

    # 3. Prediction
    #for i in range(4):
    #    sampleInputs, sampleClasses = next(iter(datasetLoader))
    #    sampleOutputs = net(cnnUtils.ToVar(sampleInputs))
    #    _, samplePreds = torch.max(sampleOutputs.data, 1)
    #    print(sampleClasses.cpu().numpy())
    #    print(samplePreds.cpu().numpy())
    #    print(sampleOutputs.cpu().data.numpy())

    datasetLen = len(datasetClasses)
    classCorrect = list(0. for i in range(datasetLen))
    classTotal = list(0. for i in range(datasetLen))

    for i, data in enumerate(testLoader):
        inputs, labels = data
        inputs, labels = cnnUtils.ToVar(inputs), cnnUtils.ToVar(labels)
    
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        classTotal[labels.data.cpu().numpy()[0]] += 1
    
        if labels.data.cpu().numpy()[0] == predicted.cpu().numpy()[0][0]:
            classCorrect[labels.data.cpu().numpy()[0]] += 1
        print(str(i) + '/' + str(datasetSize))

    for i, cls in enumerate(classCorrect):
        if classTotal[i] > 0:
            print('Class ' + datasetClasses[i] + ' total: ' + str(classTotal[i]) + ' correct: ' + str(classCorrect[i]) + ' success rate is ' + str(100 * classCorrect[i] / classTotal[i]))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
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

float_formatter = lambda x: "%.2f" % x

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind':float_formatter})

    parser = argparse.ArgumentParser()
    parser.add_argument('--resumepoint', type=str, help='checkpoint path to load')
    parser.add_argument('--startepoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--dataset', required=True, help='path of images to train and test, point to parent folder of train and test subfolders')
    parser.add_argument('--imagesize', default=128, type=int, help='images will be scaled to this size')
    parser.add_argument('--lr', default=0.0001, type=float, help='starting learning rate (if you are resuming, this value will be overwritten)')
    parser.add_argument('--numepochs', default=10, type=int, help='how many epochs for training')

    opt = parser.parse_args()

    # Should be average in the end.
    setMean = [0.5, 0.5, 0.5]
    setStd = [0.5, 0.5, 0.5]
    outputClassCount = 2
    setImageSize = opt.imagesize

    # 1. Create network
    net = DSMNLNet(setMean, setStd, setImageSize, outputClassCount)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=0.01)

    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    if opt.resumepoint:
        if os.path.isfile(opt.resumepoint):
            print("=> loading checkpoint '{}'".format(opt.resumepoint))
            checkpoint = torch.load(opt.resumepoint)
            opt.startepoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resumepoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resumepoint))

    lrScheduler = cnnUtils.StepLR(optimizer, step_size=3, gamma=0.85, last_epoch=opt.startepoch - 1)

    torch.backends.cudnn.benchmark = True

    # 2. Load the images
    if opt.dataset == '':
        print('Image path cannot be empty')


    # 2.1. Set the image transforms
    dataTransforms = {
    'train': transforms.Compose([
        transforms.Scale(setImageSize),
        transforms.RandomCrop(setImageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=setMean, std=setStd)
    ]),
    'test': transforms.Compose([
        transforms.Scale(setImageSize),
        transforms.RandomCrop(setImageSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=setMean, std=setStd)
    ]), }

    # 2.2. Create dataset and loader
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(opt.dataset, x), dataTransforms[x]) for x in ['train', 'test']}
    datasetLoaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=10, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'test']}
    testLoader = torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    datasetSizes = {x: len(datasets[x]) for x in ['train', 'test']}
    datasetClasses = datasets['train'].classes

    print(str(datasetSizes) + ' images will be used.' )
    print('GPU will ' + ('' if torch.cuda.is_available() else 'not ') + 'be used.' )
    print(str(len(datasetClasses)) + ' output classes')

    # 2.3. Load a sample batch
    sampleInputs, sampleClasses = next(iter(datasetLoaders['train']))

    # 2.4. Sanity check
    if sampleInputs.size()[0] != sampleClasses.size()[0]:
        print('Dataset is not loaded correctly. ')

    # Train 
    trainAccuracyArray = []
    testAccuracyArray = []
    lrLogArray = []
    trainErrorArray = []
    testErrorArray = []

    cnnUtils.TrainModelMiniBatch(net, criterion, optimizer, lrScheduler, datasetLoaders, datasetSizes, 
        trainAccuracyArray, testAccuracyArray, lrLogArray, trainErrorArray, testErrorArray, 
        opt.startepoch, num_epochs=opt.numepochs)


    # 3. Prediction
    #for i in range(4):
    #    sampleInputs, sampleClasses = next(iter(datasetLoader))
    #    sampleOutputs = net(cnnUtils.ToVar(sampleInputs))
    #    _, samplePreds = torch.max(sampleOutputs.data, 1)
    #    print(sampleClasses.cpu().numpy())
    #    print(samplePreds.cpu().numpy())
    #    print(sampleOutputs.cpu().data.numpy())

    # datasetLen = len(datasetClasses)
    # classCorrect = list(0. for i in range(datasetLen))
    # classTotal = list(0. for i in range(datasetLen))

    # for i, data in enumerate(testLoader):
    #     inputs, labels = data
    #     inputs, labels = cnnUtils.ToVar(inputs), cnnUtils.ToVar(labels)
    
    #     outputs = net(inputs)
    #     _, predicted = torch.max(outputs.data, 1)

    #     classTotal[labels.data.cpu().numpy()[0]] += 1
    
    #     if labels.data.cpu().numpy()[0] == predicted.cpu().numpy()[0][0]:
    #         classCorrect[labels.data.cpu().numpy()[0]] += 1
    #     print(str(i) + '/' + str(datasetSize))

    # for i, cls in enumerate(classCorrect):
    #     if classTotal[i] > 0:
    #         print('Class ' + datasetClasses[i] + ' total: ' + str(classTotal[i]) + ' correct: ' + str(classCorrect[i]) + ' success rate is ' + str(100 * classCorrect[i] / classTotal[i]))



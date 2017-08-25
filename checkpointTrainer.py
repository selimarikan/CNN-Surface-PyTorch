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
import datetime
import argparse
import cnnUtils
from neuralNets import DSMNLNet256

float_formatter = lambda x: "%.2f" % x

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind':float_formatter})

    parser = argparse.ArgumentParser()
    parser.add_argument('--resumepoint', type=str, help='checkpoint path to load')
    parser.add_argument('--startepoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--dataset', required=True, help='path of images to train and test, point to parent folder of train and test subfolders. Alternatively mnist, cifar10 values can be used.')
    parser.add_argument('--imagesize', default=128, type=int, help='images will be scaled to this size')
    parser.add_argument('--lr', default=0.0001, type=float, help='starting learning rate (if you are resuming, this value will be overwritten)')
    parser.add_argument('--numepochs', default=10, type=int, help='how many epochs for training')
    parser.add_argument('--outdim', default=2, type=int, help='output dimension of the network (class count)')
    parser.add_argument('--net', default='dsmnl', type=str, help='choose network architecture to be used. Options are: dsmnl, alexnet, resnet, densenet')

    logFileName = 'CheckpointTrainerLog.txt'
    logF = cnnUtils.TXTLogger(logFileName)
    logF.Log('------- Started CheckpointTrainer -------')
    opt = parser.parse_args()
    logF.Log(str(opt))

    # Should be average in the end.
    setMean = [0.5, 0.5, 0.5]
    setStd = [0.5, 0.5, 0.5]
    outputClassCount = opt.outdim
    setImageSize = opt.imagesize

    # 1. Create network
    if opt.net == 'dsmnl':
        net = DSMNLNet256(setMean, setStd, setImageSize, outputClassCount)
    elif opt.net == 'alexnet':
        net = torchvision.models.alexnet()
        net.classifier[6].out_features = opt.outdim
    elif opt.net == 'resnet':
        net = torchvision.models.resnet18()
        net.fc.out_features = opt.outdim
    else:
        print('Unknown network name')
        exit()

    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=0.01)

    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()

    # 2. Load the images
    if opt.dataset == '':
        print('Image path cannot be empty')
        exit()

    if opt.resumepoint:
        resumePath = os.path.join(opt.dataset, opt.resumepoint)
        if os.path.isfile(resumePath):
            print("=> loading checkpoint '{}'".format(resumePath))
            checkpoint = torch.load(resumePath)
            opt.startepoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resumePath, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resumePath))

    lrScheduler = cnnUtils.StepLR(optimizer, step_size=5, gamma=0.85, last_epoch=opt.startepoch - 1)

    torch.backends.cudnn.benchmark = True

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

    datasets = { 'train' : [], 'test': []}

    # 2.2. Create dataset and loader
    if opt.dataset == 'mnist':
        mnistTransform = transforms.Compose([
            transforms.Scale(setImageSize), 
            transforms.RandomCrop(setImageSize), 
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
        datasets['train'] = torchvision.datasets.MNIST('../MNIST', train=True, download=True, transform=mnistTransform)
        datasets['test'] = torchvision.datasets.MNIST('../MNIST', train=False, transform=mnistTransform)
        datasetClasses = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif opt.dataset == 'cifar10':
        cifar10TrainTransform = transforms.Compose([
            transforms.Scale(setImageSize),
            transforms.RandomCrop(setImageSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        cifar10TestTransform = transforms.Compose([
            transforms.Scale(setImageSize),
            transforms.RandomCrop(setImageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        datasets['train'] = torchvision.datasets.CIFAR10('../CIFAR10', train=True, download=True, transform=cifar10TrainTransform)
        datasets['test'] = torchvision.datasets.CIFAR10('../CIFAR10', train=False, transform=cifar10TestTransform)
        datasetClasses = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        datasets = {x: torchvision.datasets.ImageFolder(os.path.join(opt.dataset, x), dataTransforms[x]) for x in ['train', 'test']}
        datasetClasses = datasets['train'].classes

    datasetLoaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=10, shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'test']}
    testLoader = torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    datasetSizes = {x: len(datasets[x]) for x in ['train', 'test']}

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

    # Set the network back so that best model is used for metrics
    net = cnnUtils.TrainModelMiniBatch(net, criterion, optimizer, lrScheduler, opt.dataset, datasetLoaders, datasetSizes, 
        trainAccuracyArray, testAccuracyArray, lrLogArray, trainErrorArray, testErrorArray, 
        opt.startepoch, num_epochs=opt.numepochs)

    print('Calculating classification metrics...')

    # Print relevant statistics
    correct, total, [accuracy, precision, recall, specificity] = cnnUtils.DetermineAccuracy(net, 'test', datasetLoaders)
    statText = 'Accuracy: ' + str(accuracy) + ' Precision: ' + str(precision) + ' Recall: ' + str(recall) + ' Specificity: ' + str(specificity)
    print(statText)

    logF.Log(statText)

    logF.Log('Training set accuracy values')
    logF.Log(''.join(str(x) for x in trainAccuracyArray))
    logF.Log('Test set accuracy values')
    logF.Log(''.join(str(x) for x in testAccuracyArray))
    logF.Log('Learning rate values')
    logF.Log(''.join(str(x) for x in lrLogArray))
    logF.Log('Training error values')
    logF.Log(''.join(str(x) for x in trainErrorArray))
    logF.Log('Test error values')
    logF.Log(''.join(str(x) for x in testErrorArray))

    print('Calculating confusion matrix...')

    confusionMat = cnnUtils.CalculateConfusion(net, datasetClasses, testLoader)
    logF.Log(''.join(str(x.numpy()) for x in confusionMat))

    logF.Log('------- Finished CheckpointTrainer -------')
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



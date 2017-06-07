import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
#%matplotlib inline
plt.ion()

def ImShow(input, title=None):
    input = input.numpy().transpose((1, 2, 0))
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def TrainModelT7(model, criterion, optimizer, lrDecay, maxIteration):
    currentIteration = 0
    startTime = time.time()
    trainSetSize = datasetSizes['train']

    shuffledIndices = torch.randperm(trainSetSize)
    print('Started training...')

    while True:
        runningLoss = 0
        for iInput, data in enumerate(datasetLoaders['train'], 0):
            inputs, labels = data
            if useGPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(iInput)
            runningLoss += loss.data[0]

        currentError = runningLoss / trainSetSize
        print('Iteration: ' + str(currentIteration) + '# current error = ' + str(currentError))
        currentIteration += 1

        if maxIteration > 0 and currentIteration > maxIteration:
            print('# Maximum iteration reached. End of training.')
            print('# Training error = ' + str(currentError))
            timeElapsed = time.time() - startTime
            print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
            break


def ExpLRScheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def VisualizeModel(model, numImages=6):
    imagesSoFar = 0
    fig = plt.figure()

    for i, data in enumerate(datasetLoaders['test']):
        inputs, labels = data
        if useGPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            imagesSoFar += 1
            ax = plt.subplot(numImages // 2, 2, imagesSoFar)
            ax.axis('off')
            ax.set_title('Predicted: {}'.format(datasetClasses[labels.data[j]]))
            ImShow(inputs.cpu().data[j])

            if imagesSoFar == numImages:
                return


# outMul = setImageSize / 8

# model = nn.Sequential(
#    nn.Conv2d(1, 32, (3,3), (1,1), (1,1)),
#    nn.ReLU(),
#    nn.MaxPool2d(2, 2),
#    nn.Conv2d(32, 32, (3,3), (1,1), (1,1)),
#    nn.ReLU(),
#    nn.MaxPool2d(2, 2),
#    nn.Conv2d(32, 32, (3,3), (1,1), (1,1)),
#    nn.ReLU(),
#    nn.MaxPool2d(2, 2),
# )


# Currently, weights are not shared.
# TODO: Try weight sharing with looping the element
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.pool3 = nn.MaxPool2d(2, 2)

        # requires int casting in Python, might run into problems with other image sizes
        self.outMul = int(setImageSize / 8)
        self.fc1 = nn.Linear(32 * self.outMul * self.outMul, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2)  # number of classes
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * self.outMul * self.outMul)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = F.dropout(F.relu(self.fc3(x)), 0.5)
        x = self.logsmax(self.fc4(x))
        return x

if __name__ == '__main__':
    dataTransforms = {
        'train': transforms.Compose([
            transforms.Scale((64, 64)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Scale((64, 64)),
            transforms.ToTensor()
        ]), }

    baseDirectory = 'c:/Users/Selim/Documents/GitHub/Files/'
    setDirectory = '3MSet_Large_Py'
    setImageSize = 64
    setPath = os.path.join(baseDirectory, setDirectory)
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(setPath, x), dataTransforms[x])
                for x in ['train', 'test']}
    datasetLoaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=True, num_workers=4)
                      for x in ['train', 'test']}
    datasetSizes = {x: len(datasets[x]) for x in ['train', 'test']}
    datasetClasses = datasets['train'].classes

    useGPU = torch.cuda.is_available()
    print(str(datasetSizes) + ' images will be used')
    print('GPU will ' + ('' if useGPU else 'not ') + 'be used')

    inputs, classes = next(iter(datasetLoaders['train']))
    out = torchvision.utils.make_grid(inputs)
    #ImShow(out, title=[datasetClasses[x] for x in classes])

    net = Net()
    if useGPU:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if useGPU:
        criterion = criterion.cuda()

    TrainModelT7(net, criterion, optimizer, 0, maxIteration=5)
    VisualizeModel(net)

    correct = 0
    total = 0
    for data in datasetLoaders['test']:
        # Inputs
        inputs, labels = data
        if useGPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('Accuracy of the network is {}%'.format(100 * correct / total))

    classCorrect = list(0. for i in range(2))
    classTotal = list(0. for i in range(2))

    for data in datasetLoaders['test']:
        inputs, labels = data
        if useGPU:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print('Ground truth: ' + str(labels.data.numpy()[0]) + ' predicted: ' + str(predicted.numpy()[0][0]))
        classTotal[labels.data.numpy()[0]] += 1

        if labels.data.numpy()[0] == predicted.numpy()[0][0]:
            classCorrect[labels.data.numpy()[0]] += 1

    for i, cls in enumerate(classCorrect):
        print('Class {} total {}: success rate is {}'.format(i, classTotal[i], (100 * classCorrect[i] / classTotal[i])))





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
import cnnUtils


def CreateNet(netName, setMean, setStd, setImageSize, outputClassCount):
    net = None
    criterion = None
    
    if netName == 'dsmnl':
        net = DSMNLNet256(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv2':
        net = DSMNLNet256v2(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv3':
        net = DSMNLNet128v3(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv4':
        net = DSMNLNet128v4(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv4ar':
        net = DSMNLNet128v4ar(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv4b':
        net = DSMNLNet128v4b(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'dsmnlv5':
        net = DSMNLNet128v5(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()
    
    elif netName == 'channelnet':
        net = ChannelNet(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'experimental':
        net = ExperimentalNet(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'buzz':
        net = BuzzNet(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()
    
    elif netName == 'buzzv2':
        net = BuzzNetv2(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'buzzv3':
        net = BuzzNetv3(setMean, setStd, setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'bnoptim':
        net = NetBNOptim(setImageSize, outputClassCount)
        criterion = nn.NLLLoss()

    elif netName == 'alexnet':
        net = torchvision.models.alexnet()
        net.classifier[6].out_features = opt.outdim
        criterion = nn.CrossEntropyLoss()

    elif netName == 'vgg':
        net = torchvision.models.vgg16_bn(num_classes=opt.outdim)
        criterion = nn.CrossEntropyLoss()

    elif netName == 'resnet':
        net = torchvision.models.resnet18(pretrained=True)
        net.fc.out_features = opt.outdim
        criterion = nn.CrossEntropyLoss()

    elif netName == 'resnet152pt':
        net = torchvision.models.resnet152(pretrained=True)
        net.fc.out_features = opt.outdim
        criterion = nn.CrossEntropyLoss()
    
    elif netName == 'densenet':
        net = torchvision.models.densenet121()
        net.fc.out_features = opt.outdim
        criterion = nn.CrossEntropyLoss()
    
    elif netName == 'aec':
        net = AEC(setImageSize)
        criterion = nn.L1Loss()

    else:
        print('Unknown network name')

    return net, criterion

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.reluIn = nn.ReLU()
        self.poolIn = nn.MaxPool2d(2, 2)
        
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.reluI2 = nn.ReLU()
        self.poolI2 = nn.MaxPool2d(2, 2)
        
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.reluI3 = nn.ReLU()
        self.poolI3 = nn.MaxPool2d(2, 2)
        
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, len(datasetClasses))) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        x = self.poolIn(self.reluIn(convInResult))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(cnnUtils.save_grad('convI2Grad'))
        x = self.poolI2(self.reluI2(convI2Result))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(cnnUtils.save_grad('convI3Grad'))
        x = self.poolI3(self.reluI3(convI3Result))
        
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class NetBN(nn.Module):
    def __init__(self):
        super(NetBN, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
        self.poolIn = nn.MaxPool2d(2, 2)
        
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.ReLU()
        self.poolI2 = nn.MaxPool2d(2, 2)
        
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.reluI3 = nn.ReLU()
        self.poolI3 = nn.MaxPool2d(2, 2)
        
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, len(datasetClasses))) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        x = self.poolIn(self.reluIn(self.batNIn(convInResult)))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(cnnUtils.save_grad('convI2Grad'))
        x = self.poolI2(self.reluI2(self.batNI2(convI2Result)))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(cnnUtils.save_grad('convI3Grad'))
        x = self.poolI3(self.reluI3(self.batNI3(convI3Result)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class NetBNSELU(nn.Module):
    def __init__(self):
        super(NetBNSELU, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=32)
        self.seluIn = SELU()
        self.poolIn = nn.MaxPool2d(2, 2)
        
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.seluI2 = SELU()
        self.poolI2 = nn.MaxPool2d(2, 2)
        
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.seluI3 = SELU()
        self.poolI3 = nn.MaxPool2d(2, 2)
        
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, 4096),
            SELU(),
            AlphaDropout(p=0.5),
            nn.Linear(4096, 4096),
            SELU(),
            AlphaDropout(p=0.5),
            nn.Linear(4096, 4096),
            SELU(),
            AlphaDropout(p=0.5),
            nn.Linear(4096, len(datasetClasses))) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        x = self.poolIn(self.seluIn(self.batNIn(convInResult)))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(cnnUtils.save_grad('convI2Grad'))
        x = self.poolI2(self.seluI2(self.batNI2(convI2Result)))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(cnnUtils.save_grad('convI3Grad'))
        x = self.poolI3(self.seluI3(self.batNI3(convI3Result)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class NetBNOptim(nn.Module):
    def __init__(self, setImageSize, outputClassCount):
        super(NetBNOptim, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
        self.poolIn = nn.MaxPool2d(2, 2)
        
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.batNI2 = nn.BatchNorm2d(num_features=64)
        self.reluI2 = nn.ReLU()
        self.poolI2 = nn.MaxPool2d(2, 2)
        
        # Out CH = 32 !
        self.convI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.reluI3 = nn.ReLU()
        self.poolI3 = nn.MaxPool2d(2, 2)
        
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1024, outputClassCount)) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        x = self.poolIn(self.reluIn(self.batNIn(convInResult)))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        x = self.poolI2(self.reluI2(self.batNI2(convI2Result)))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        x = self.poolI3(self.reluI3(self.batNI3(convI3Result)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x



def alpha_dropout(input, p=0.5, training=False):
    r"""Applies alpha dropout to the input.
    See :class:`~torch.nn.AlphaDropout` for details.
    Args:
        p (float, optional): the drop probability
        training (bool, optional): switch between training and evaluation mode
    """
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))

    if p == 0 or not training:
        return input

    alpha = -1.7580993408473766
    keep_prob = 1 - p
    # TODO avoid casting to byte after resize
    noise = input.data.new().resize_(input.size())
    noise.bernoulli_(p)
    noise = Variable(noise.byte())

    output = input.masked_fill(noise, alpha)

    a = (keep_prob + alpha ** 2 * keep_prob * (1 - keep_prob)) ** (-0.5)
    b = -a * alpha * (1 - keep_prob)

    return output.mul_(a).add_(b)

class AlphaDropout(nn.Module):
    r"""Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        p (float): probability of an element to be dropped. Default: 0.5

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.AlphaDropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, p=0.5):
        super(AlphaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return alpha_dropout(input, self.p, self.training)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) + ')'

    
class SELU_THNN(torch.autograd.function.InplaceFunction):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def forward(ctx, input, inplace):
        backend = torch._thnn.type2backend[type(input)]
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.new(input.size())
        backend.ELU_updateOutput(
            backend.library_state,
            input,
            output,
            SELU_THNN.alpha,
            inplace,
        )
        output.mul_(SELU_THNN.scale)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        if grad_output.volatile:
            grad_input = Variable(input.data.new(input.size()), volatile=True)
            backend = torch._thnn.type2backend[type(input.data)]
            backend.ELU_updateGradInput(
                backend.library_state,
                input.data,
                grad_output.data.mul(SELU_THNN.scale),
                grad_input.data,
                output.data.div(SELU_THNN.scale),
                SELU_THNN.alpha,
                False
            )
        else:
            positive_mask = (output > 0).type_as(grad_output)
            negative_mask = (output <= 0).type_as(grad_output)
            grad_input = grad_output * SELU_THNN.scale * (positive_mask +
                                                     negative_mask * (output / SELU_THNN.scale + SELU_THNN.alpha))
        return grad_input, None

def selu(input, inplace=False):
    return SELU_THNN.apply(input, inplace)

class SELU(nn.Module):
    """Applies element-wise, :math:`f(x) = scale * (\max(0,x) + \min(0, alpha * (\exp(x) - 1)))`,
    with ``alpha=1.6732632423543772848170429916717`` and ``scale=1.0507009873554804934193349852946``.
    More details can be found in the paper `Self-Normalizing Neural Networks`_ .
    Args:
        inplace (bool, optional): can optionally do the operation in-place
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.SELU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return selu(input, self.inplace)

    def __repr__(self):
        inplace_str = ' (inplace)' if self.inplace else ''
        return self.__class__.__name__ + inplace_str

# Dilate Stride Non-Linearity Network
class DSMNLNet32(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet32, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
        
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=64)
        self.reluI2 = nn.ReLU()
        
        # Out CH = 32 !
        self.convI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.reluI3 = nn.ReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=32)
        self.reluI4 = nn.ReLU()
        
        self.convI5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=32)
        self.reluI5 = nn.ReLU()
        
        self.convI6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=32)
        self.reluI6 = nn.ReLU()
        
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, outputClassCount))
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            #nn.Linear(4096, len(datasetClasses))) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        x = self.reluI5(self.batNI5(self.convI5(x)))
        x = self.reluI6(self.batNI6(self.convI6(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

# Dilate Stride Non-Linearity Network
class DSMNLNet256(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet256, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=64)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIA = nn.BatchNorm2d(num_features=64)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=128)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIB = nn.BatchNorm2d(num_features=128)
        self.reluIB = nn.PReLU()
        
        # Out CH = 256 !
        self.convI3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=256)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIC = nn.BatchNorm2d(num_features=256)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=256)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=256)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=256)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(256 * self.outMul * self.outMul, outputClassCount),
            #nn.ReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(4096, len(datasetClasses))) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        
        x = self.reluIA(self.batNIA(self.convIA(x)))
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        
        x = self.reluIB(self.batNIB(self.convIB(x)))
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        
        x = self.reluIC(self.batNIC(self.convIC(x)))
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        x = self.reluI5(self.batNI5(self.convI5(x)))
        x = self.reluI6(self.batNI6(self.convI6(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 256 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

# Dilate Stride Non-Linearity Network
class DSMNLNet256v2(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet256v2, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=64)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIA = nn.BatchNorm2d(num_features=64)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=128)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIB = nn.BatchNorm2d(num_features=128)
        self.reluIB = nn.PReLU()
        
        # Out CH = 256 !
        self.convI3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=256)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIC = nn.BatchNorm2d(num_features=256)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=256)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=256)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=256)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(256 * self.outMul * self.outMul, outputClassCount),
            #nn.ReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(4096, len(datasetClasses))) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + resIn)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + resI2)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        x = self.reluI5(self.batNI5(self.convI5(x)))
        x = self.reluI6(self.batNI6(self.convI6(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 256 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

# Dilate Stride Non-Linearity Network
class DSMNLNet128v3(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet128v3, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=16)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIB = nn.BatchNorm2d(num_features=32)
        self.reluIB = nn.PReLU()
        
        # Out CH = 128 !
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=64)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIC = nn.BatchNorm2d(num_features=64)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=64)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=64)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=64)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + resIn)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + resI2)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

# Dilate Stride Non-Linearity Network
class DSMNLNet128v4(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet128v4, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=16)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIB = nn.BatchNorm2d(num_features=32)
        self.reluIB = nn.PReLU()
        
        # Out CH = 128 !
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=64)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIC = nn.BatchNorm2d(num_features=64)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI4 = nn.BatchNorm2d(num_features=64)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI5 = nn.BatchNorm2d(num_features=64)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI6 = nn.BatchNorm2d(num_features=64)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + resIn)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + resI2)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class DSMNLNet128v4ar(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet128v4ar, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=16)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIB = nn.BatchNorm2d(num_features=32)
        self.reluIB = nn.PReLU()
        
        # Out CH = 128 !
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=64)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIC = nn.BatchNorm2d(num_features=64)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI4 = nn.BatchNorm2d(num_features=64)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI5 = nn.BatchNorm2d(num_features=64)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI6 = nn.BatchNorm2d(num_features=64)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def GetMatchingLayer(self, inCh, outCh, stride, x):
        c2d = nn.Conv2d(inCh, outCh, kernel_size=1, stride=stride, bias=False).cuda()
        bn = nn.BatchNorm2d(outCh).cuda()
        return bn(c2d(x))

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        plus01 = self.GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels, self.convIn.stride, x)
        
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult))# + plus01)
        #resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + plus01)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        plus02 = self.GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels, self.convI2.stride, x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result))# + plus02)
        #resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + plus02)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        plus03 = self.GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels, self.convI3.stride, x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        #resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + plus03)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

# Dilate Stride Non-Linearity Network
class DSMNLNet128v4b(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet128v4b, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=16)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIB = nn.BatchNorm2d(num_features=32)
        self.reluIB = nn.PReLU()
        
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIC = nn.BatchNorm2d(num_features=32)
        self.reluIC = nn.PReLU()
        
        self.convIX = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNIX = nn.BatchNorm2d(num_features=64)
        self.reluIX = nn.PReLU()
        
        self.convIY = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIY = nn.BatchNorm2d(num_features=64)
        self.reluIY = nn.PReLU()

        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI4 = nn.BatchNorm2d(num_features=64)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI5 = nn.BatchNorm2d(num_features=64)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI6 = nn.BatchNorm2d(num_features=64)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 16) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + resIn)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + resI2)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)

        x = self.reluIX(self.batNIX(self.convIX(x)))
        resIX = x
        x = self.reluIY(self.batNIY(self.convIY(x)) + resIX)
        resIY = x
        x = self.reluI4(self.batNI4(self.convI4(x)) + resIY)
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        #print (x.size())
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class DSMNLNet128v5(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet128v5, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIB = nn.BatchNorm2d(num_features=16)
        self.reluIB = nn.PReLU()
        
        self.convI3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=32)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIC = nn.BatchNorm2d(num_features=32)
        self.reluIC = nn.PReLU()
        
        self.convIX = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNIX = nn.BatchNorm2d(num_features=64)
        self.reluIX = nn.PReLU()
        
        self.convIY = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIY = nn.BatchNorm2d(num_features=64)
        self.reluIY = nn.PReLU()

        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI4 = nn.BatchNorm2d(num_features=32)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI5 = nn.BatchNorm2d(num_features=32)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI6 = nn.BatchNorm2d(num_features=32)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 16) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)))# + resIn)
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)))# + resI2)
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)

        x = self.reluIX(self.batNIX(self.convIX(x)))
        resIX = x
        x = self.reluIY(self.batNIY(self.convIY(x)) + resIX)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        #print (x.size())
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x


# Dilate Stride Residual (Later HF, MF, LF) Non-Linearity Network
class DSRNLNet256(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet256, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=64)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIA = nn.BatchNorm2d(num_features=64)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=128)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIB = nn.BatchNorm2d(num_features=64)
        self.reluIB = nn.PReLU()
        
        # Out CH = 256 !
        self.convI3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=256)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIC = nn.BatchNorm2d(num_features=256)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=256)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=256)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=256)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(256 * self.outMul * self.outMul, outputClassCount),
            #nn.ReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(4096, len(datasetClasses))) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        
        x = self.reluIA(self.batNIA(self.convIA(x)))
        
        # Convolutional layer 2
        #  Get the residual, match it to the output
        #plus02 = GetMatchingLayer(self.convI2.in_channels, self.convI2.out_channels)(x)
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        
        x = self.reluIB(self.batNIB(self.convIB(x)))
        
        # Convolutional layer 3
        #  Get the residual, match it to the output
        #plus03 = GetMatchingLayer(self.convI3.in_channels, self.convI3.out_channels)(x)
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        
        x = self.reluIC(self.batNIC(self.convIC(x)))
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        x = self.reluI5(self.batNI5(self.convI5(x)))
        x = self.reluI6(self.batNI6(self.convI6(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 256 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class ChannelNet(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(ChannelNet, self).__init__()

        # High-pass lane
        self.convoHPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrHPIn = nn.BatchNorm2d(num_features=32)
        self.preluHPIn = nn.PReLU()
        
        self.convoHPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrHPIA = nn.BatchNorm2d(num_features=32)
        self.preluHPIA = nn.PReLU()
                
        self.convoHPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrHPI2 = nn.BatchNorm2d(num_features=64)
        self.preluHPI2 = nn.PReLU()
        
        self.convoHPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrHPIB = nn.BatchNorm2d(num_features=64)
        self.preluHPIB = nn.PReLU()
        
        self.convoHPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrHPI3 = nn.BatchNorm2d(num_features=32)
        self.preluHPI3 = nn.PReLU()
        
        # Band-pass lane
        self.convoBPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNrBPIn = nn.BatchNorm2d(num_features=32)
        self.preluBPIn = nn.PReLU()
        
        self.convoBPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrBPIA = nn.BatchNorm2d(num_features=32)
        self.preluBPIA = nn.PReLU()
                
        self.convoBPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNrBPI2 = nn.BatchNorm2d(num_features=64)
        self.preluBPI2 = nn.PReLU()
        
        self.convoBPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrBPIB = nn.BatchNorm2d(num_features=64)
        self.preluBPIB = nn.PReLU()
        
        self.convoBPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrBPI3 = nn.BatchNorm2d(num_features=32)
        self.preluBPI3 = nn.PReLU()

        # Low-pass lane
        self.convoLPIn = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrLPIn = nn.BatchNorm2d(num_features=64)
        self.preluLPIn = nn.PReLU()
        
        self.convoLPIA = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrLPIA = nn.BatchNorm2d(num_features=64)
        self.preluLPIA = nn.PReLU()
                
        self.maxPoolI2 = nn.MaxPool2d(2, 2)
        
        self.convoLPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrLPI3 = nn.BatchNorm2d(num_features=32)
        self.preluLPI3 = nn.PReLU()
        
        # Extended nonlinearity
        self.convoAPI4 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrAPI4 = nn.BatchNorm2d(num_features=64)
        self.preluAPI4 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def RunHP(self, x):
        ## HP lane
        xhp = x
        # Convolutional layer 1
        convoHPInResult = self.convoHPIn(xhp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xhp = self.preluHPIn(self.batNrHPIn(convoHPInResult)) #  + plus01
        resHPIn = xhp
        
        xhp = self.preluHPIA(self.batNrHPIA(self.convoHPIA(xhp)) + resHPIn)
        
        # Convolutional layer 2
        convoHPI2Result = self.convoHPI2(xhp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI2(self.batNrHPI2(convoHPI2Result)) # + plus02
        resHPI2 = xhp

        xhp = self.preluHPIB(self.batNrHPIB(self.convoHPIB(xhp)) + resHPI2)
        
        # Convolutional layer 3
        convoHPI3Result = self.convoHPI3(xhp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI3(self.batNrHPI3(convoHPI3Result)) #  + plus03

        return xhp

    def RunBP(self, x):
        ## BP lane
        xbp = x
        # Convolutional layer 1
        convoBPInResult = self.convoBPIn(xbp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xbp = self.preluBPIn(self.batNrBPIn(convoBPInResult)) #  + plus01
        resBPIn = xbp
        
        xbp = self.preluBPIA(self.batNrBPIA(self.convoBPIA(xbp)) + resBPIn)
        
        # Convolutional layer 2
        convoBPI2Result = self.convoBPI2(xbp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI2(self.batNrBPI2(convoBPI2Result)) # + plus02
        resBPI2 = xbp

        xbp = self.preluBPIB(self.batNrBPIB(self.convoBPIB(xbp)) + resBPI2)
        
        # Convolutional layer 3
        convoBPI3Result = self.convoBPI3(xbp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI3(self.batNrBPI3(convoBPI3Result)) #  + plus03

        return xbp

    def RunLP(self, x):
        ## LP lane
        xlp = x
        # Convolutional layer 1
        convoLPInResult = self.convoLPIn(xlp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xlp = self.preluLPIn(self.batNrLPIn(convoLPInResult)) #  + plus01
        resLPIn = xlp
        
        xlp = self.preluLPIA(self.batNrLPIA(self.convoLPIA(xlp)) + resLPIn)
        
        xlp = self.maxPoolI2(xlp)

        # Convolutional layer 3
        convoLPI3Result = self.convoLPI3(xlp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xlp = self.preluLPI3(self.batNrLPI3(convoLPI3Result)) #  + plus03

        return xlp

    def forward(self, x):
        
        xhp = self.RunHP(x)

        xbp = self.RunBP(x)

        xlp = self.RunLP(x)

        x = [xhp, xbp, xlp]
        x = torch.cat(x, 1)

        x = self.preluAPI4(self.batNrAPI4(self.convoAPI4(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class ChannelNetv2(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(ChannelNetv2, self).__init__()

        # High-pass lane
        self.convoHPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrHPIn = nn.BatchNorm2d(num_features=32)
        self.preluHPIn = nn.PReLU()
        
        self.convoHPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrHPIA = nn.BatchNorm2d(num_features=32)
        self.preluHPIA = nn.PReLU()
                
        self.convoHPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrHPI2 = nn.BatchNorm2d(num_features=64)
        self.preluHPI2 = nn.PReLU()
        
        self.convoHPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrHPIB = nn.BatchNorm2d(num_features=64)
        self.preluHPIB = nn.PReLU()
        
        self.convoHPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrHPI3 = nn.BatchNorm2d(num_features=32)
        self.preluHPI3 = nn.PReLU()
        
        # Band-pass lane
        self.convoBPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNrBPIn = nn.BatchNorm2d(num_features=32)
        self.preluBPIn = nn.PReLU()
        
        self.convoBPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrBPIA = nn.BatchNorm2d(num_features=32)
        self.preluBPIA = nn.PReLU()
                
        self.convoBPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNrBPI2 = nn.BatchNorm2d(num_features=64)
        self.preluBPI2 = nn.PReLU()
        
        self.convoBPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrBPIB = nn.BatchNorm2d(num_features=64)
        self.preluBPIB = nn.PReLU()
        
        self.convoBPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrBPI3 = nn.BatchNorm2d(num_features=32)
        self.preluBPI3 = nn.PReLU()

        # Low-pass lane
        self.convoLPIn = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrLPIn = nn.BatchNorm2d(num_features=64)
        self.preluLPIn = nn.PReLU()
        
        self.convoLPIA = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrLPIA = nn.BatchNorm2d(num_features=64)
        self.preluLPIA = nn.PReLU()
                
        self.maxPoolI2 = nn.MaxPool2d(2, 2)
        
        self.convoLPI3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.batNrLPI3 = nn.BatchNorm2d(num_features=32)
        self.preluLPI3 = nn.PReLU()
        
        # Extended nonlinearity
        self.convoAPI4 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNrAPI4 = nn.BatchNorm2d(num_features=64)
        self.preluAPI4 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def RunHP(self, x):
        ## HP lane
        xhp = x
        # Convolutional layer 1
        convoHPInResult = self.convoHPIn(xhp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xhp = self.preluHPIn(self.batNrHPIn(convoHPInResult)) #  + plus01
        resHPIn = xhp
        
        xhp = self.preluHPIA(self.batNrHPIA(self.convoHPIA(xhp)) + resHPIn)
        
        # Convolutional layer 2
        convoHPI2Result = self.convoHPI2(xhp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI2(self.batNrHPI2(convoHPI2Result)) # + plus02
        resHPI2 = xhp

        xhp = self.preluHPIB(self.batNrHPIB(self.convoHPIB(xhp)) + resHPI2)
        
        # Convolutional layer 3
        convoHPI3Result = self.convoHPI3(xhp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI3(self.batNrHPI3(convoHPI3Result)) #  + plus03

        return xhp

    def RunBP(self, x):
        ## BP lane
        xbp = x
        # Convolutional layer 1
        convoBPInResult = self.convoBPIn(xbp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xbp = self.preluBPIn(self.batNrBPIn(convoBPInResult)) #  + plus01
        resBPIn = xbp
        
        xbp = self.preluBPIA(self.batNrBPIA(self.convoBPIA(xbp)) + resBPIn)
        
        # Convolutional layer 2
        convoBPI2Result = self.convoBPI2(xbp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI2(self.batNrBPI2(convoBPI2Result)) # + plus02
        resBPI2 = xbp

        xbp = self.preluBPIB(self.batNrBPIB(self.convoBPIB(xbp)) + resBPI2)
        
        # Convolutional layer 3
        convoBPI3Result = self.convoBPI3(xbp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI3(self.batNrBPI3(convoBPI3Result)) #  + plus03

        return xbp

    def RunLP(self, x):
        ## LP lane
        xlp = x
        # Convolutional layer 1
        convoLPInResult = self.convoLPIn(xlp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xlp = self.preluLPIn(self.batNrLPIn(convoLPInResult)) #  + plus01
        resLPIn = xlp
        
        xlp = self.preluLPIA(self.batNrLPIA(self.convoLPIA(xlp)) + resLPIn)
        
        xlp = self.maxPoolI2(xlp)

        # Convolutional layer 3
        convoLPI3Result = self.convoLPI3(xlp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xlp = self.preluLPI3(self.batNrLPI3(convoLPI3Result)) #  + plus03

        return xlp

    def forward(self, x):
        
        xhp = self.RunHP(x)

        xbp = self.RunBP(x)

        xlp = self.RunLP(x)

        x = [xhp, xbp, xlp]
        x = torch.cat(x, 1)

        x = self.preluAPI4(self.batNrAPI4(self.convoAPI4(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

#
class ExperimentalNet(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(ExperimentalNet, self).__init__()

        # High-pass lane
        self.convoHPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNrHPIn = nn.BatchNorm2d(num_features=32)
        self.preluHPIn = nn.PReLU()
        
        self.convoHPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrHPIA = nn.BatchNorm2d(num_features=32)
        self.preluHPIA = nn.PReLU()
                
        self.convoHPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNrHPI2 = nn.BatchNorm2d(num_features=64)
        self.preluHPI2 = nn.PReLU()
        
        self.convoHPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrHPIB = nn.BatchNorm2d(num_features=64)
        self.preluHPIB = nn.PReLU()
        
        self.convoHPI3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNrHPI3 = nn.BatchNorm2d(num_features=128)
        self.preluHPI3 = nn.PReLU()
        
        # Band-pass lane
        self.convoBPIn = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrBPIn = nn.BatchNorm2d(num_features=32)
        self.preluBPIn = nn.PReLU()
        
        self.convoBPIA = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrBPIA = nn.BatchNorm2d(num_features=32)
        self.preluBPIA = nn.PReLU()
                
        self.convoBPI2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrBPI2 = nn.BatchNorm2d(num_features=64)
        self.preluBPI2 = nn.PReLU()
        
        self.convoBPIB = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrBPIB = nn.BatchNorm2d(num_features=64)
        self.preluBPIB = nn.PReLU()
        
        self.convoBPI3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrBPI3 = nn.BatchNorm2d(num_features=128)
        self.preluBPI3 = nn.PReLU()

        # Low-pass lane
        self.convoLPIn = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrLPIn = nn.BatchNorm2d(num_features=64)
        self.preluLPIn = nn.PReLU()
        
        self.convoLPIA = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrLPIA = nn.BatchNorm2d(num_features=64)
        self.preluLPIA = nn.PReLU()
                
        self.maxPoolI2 = nn.MaxPool2d(2, 2)
        
        self.convoLPI3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(3,3), dilation=(3,3))
        self.batNrLPI3 = nn.BatchNorm2d(num_features=128)
        self.preluLPI3 = nn.PReLU()
        
        # Extended nonlinearity
        self.convoAPI4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNrAPI4 = nn.BatchNorm2d(num_features=256)
        self.preluAPI4 = nn.PReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(256 * self.outMul * self.outMul, outputClassCount),
            )
        self.logsmax = nn.LogSoftmax()

    def RunHP(self, x):
        ## HP lane
        xhp = x
        # Convolutional layer 1
        convoHPInResult = self.convoHPIn(xhp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xhp = self.preluHPIn(self.batNrHPIn(convoHPInResult)) #  + plus01
        resHPIn = xhp
        
        xhp = self.preluHPIA(self.batNrHPIA(self.convoHPIA(xhp)) + resHPIn)
        
        # Convolutional layer 2
        convoHPI2Result = self.convoHPI2(xhp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI2(self.batNrHPI2(convoHPI2Result)) # + plus02
        resHPI2 = xhp

        xhp = self.preluHPIB(self.batNrHPIB(self.convoHPIB(xhp)) + resHPI2)
        
        # Convolutional layer 3
        convoHPI3Result = self.convoHPI3(xhp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xhp = self.preluHPI3(self.batNrHPI3(convoHPI3Result)) #  + plus03

        return xhp

    def RunBP(self, x):
        ## BP lane
        xbp = x
        # Convolutional layer 1
        convoBPInResult = self.convoBPIn(xbp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xbp = self.preluBPIn(self.batNrBPIn(convoBPInResult)) #  + plus01
        resBPIn = xbp
        
        xbp = self.preluBPIA(self.batNrBPIA(self.convoBPIA(xbp)) + resBPIn)
        
        # Convolutional layer 2
        convoBPI2Result = self.convoBPI2(xbp)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI2(self.batNrBPI2(convoBPI2Result)) # + plus02
        resBPI2 = xbp

        xbp = self.preluBPIB(self.batNrBPIB(self.convoBPIB(xbp)) + resBPI2)
        
        # Convolutional layer 3
        convoBPI3Result = self.convoBPI3(xbp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xbp = self.preluBPI3(self.batNrBPI3(convoBPI3Result)) #  + plus03

        return xbp

    def RunLP(self, x):
        ## LP lane
        xlp = x
        # Convolutional layer 1
        convoLPInResult = self.convoLPIn(xlp)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        xlp = self.preluLPIn(self.batNrLPIn(convoLPInResult)) #  + plus01
        resLPIn = xlp
        
        xlp = self.preluLPIA(self.batNrLPIA(self.convoLPIA(xlp)) + resLPIn)
        
        xlp = self.maxPoolI2(xlp)

        # Convolutional layer 3
        convoLPI3Result = self.convoLPI3(xlp)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        xlp = self.preluLPI3(self.batNrLPI3(convoLPI3Result)) #  + plus03

        return xlp

    def forward(self, x):
        
        xhp = self.RunHP(x)

        xbp = self.RunBP(x)

        xlp = self.RunLP(x)

        x = [xhp, xbp, xlp]
        x = torch.cat(x, 1)

        x = self.preluAPI4(self.batNrAPI4(self.convoAPI4(x)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 256 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class BuzzNet(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(BuzzNet, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(4,4), padding=(1,1), dilation=(1,1))
        #self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
            
        self.outMul = int(setImageSize / 4) 
            
        self.fc = nn.Sequential(
            nn.Linear(32 * self.outMul * self.outMul, outputClassCount),
            #nn.PReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(1024, outputClassCount) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(convInResult)#self.batNIn(convInResult)) #  + plus01

        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class BuzzNetv2(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(BuzzNetv2, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        #self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
            
        self.outMul = int(setImageSize / 4) 
            
        self.fc = nn.Sequential(
            nn.Linear(16 * self.outMul * self.outMul, outputClassCount),
            #nn.PReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(1024, outputClassCount) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.convI2(convInResult))#self.batNIn(convInResult)) #  + plus01

        # Reshape the result for fully-connected layers
        x = x.view(-1, 16 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class BuzzNetv3(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(BuzzNetv3, self).__init__()
        self.convIn = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        self.convI3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1))
        #self.batNIn = nn.BatchNorm2d(num_features=32)
        self.reluIn = nn.ReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(16 * self.outMul * self.outMul, outputClassCount),
            #nn.PReLU(), # Enabling ReLU makes network stuck at some level
            #nn.Dropout(p=0.5),
            #nn.Linear(1024, outputClassCount) # number of classes
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        #  Get the residual, match it to the output
        #plus01 = GetMatchingLayer(self.convIn.in_channels, self.convIn.out_channels)(x)
        convInResult = self.convIn(x)
        #convInResult.register_hook(save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.convI3(self.convI2(convInResult)))#self.batNIn(convInResult)) #  + plus01

        # Reshape the result for fully-connected layers
        x = x.view(-1, 16 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class AEC(nn.Module):
    def __init__(self, setImageSize):
        super(AEC, self).__init__()
        self.setImageSize = setImageSize
        self.fc1 = nn.Linear(setImageSize * setImageSize * 3, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, setImageSize * setImageSize * 3)
        #self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        
        x = x.view(-1, self.setImageSize * self.setImageSize * 3)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
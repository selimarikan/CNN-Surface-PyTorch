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
        convInResult.register_hook(save_grad('convInGrad'))
        x = self.poolIn(self.reluIn(convInResult))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(save_grad('convI2Grad'))
        x = self.poolI2(self.reluI2(convI2Result))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(save_grad('convI3Grad'))
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
        convInResult.register_hook(save_grad('convInGrad'))
        x = self.poolIn(self.seluIn(self.batNIn(convInResult)))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(save_grad('convI2Grad'))
        x = self.poolI2(self.seluI2(self.batNI2(convI2Result)))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(save_grad('convI3Grad'))
        x = self.poolI3(self.seluI3(self.batNI3(convI3Result)))
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 32 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x

class NetBNOptim(nn.Module):
    def __init__(self):
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
            nn.Linear(1024, len(datasetClasses))) # number of classes
        
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        convInResult.register_hook(save_grad('convInGrad'))
        x = self.poolIn(self.reluIn(self.batNIn(convInResult)))
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        convI2Result.register_hook(save_grad('convI2Grad'))
        x = self.poolI2(self.reluI2(self.batNI2(convI2Result)))
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        convI3Result.register_hook(save_grad('convI3Grad'))
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

class DSMNLNet256(nn.Module):
    def __init__(self, mean, std, setImageSize, outputClassCount):
        super(DSMNLNet256, self).__init__()
        self.convIn = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNIn = nn.BatchNorm2d(num_features=64)
        self.reluIn = nn.ReLU()
        
        self.convIA = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIA = nn.BatchNorm2d(num_features=64)
        self.reluIA = nn.ReLU()
                
        self.convI2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI2 = nn.BatchNorm2d(num_features=128)
        self.reluI2 = nn.ReLU()
        
        self.convIB = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIB = nn.BatchNorm2d(num_features=128)
        self.reluIB = nn.ReLU()
        
        # Out CH = 256 !
        self.convI3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(6,6), dilation=(3,3))
        self.batNI3 = nn.BatchNorm2d(num_features=256)
        self.reluI3 = nn.ReLU()
        
        self.convIC = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNIC = nn.BatchNorm2d(num_features=256)
        self.reluIC = nn.ReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI4 = nn.BatchNorm2d(num_features=256)
        self.reluI4 = nn.ReLU()
        
        self.convI5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI5 = nn.BatchNorm2d(num_features=256)
        self.reluI5 = nn.ReLU()
        
        self.convI6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(3,3))
        self.batNI6 = nn.BatchNorm2d(num_features=256)
        self.reluI6 = nn.ReLU()
            
        self.outMul = int(setImageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(256 * self.outMul * self.outMul, outputClassCount))
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
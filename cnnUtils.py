import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Spectral7
from bokeh.io import output_notebook
#import visdom
#from graphviz import Digraph
import numpy as np
import datetime
import gc
import time
import math
import copy
import os
import PIL
plt.ion()

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def ImShow(input, mean, std, title=None):
    input = input.numpy().transpose((1, 2, 0))
    #input = std * input + mean
    plt.figure(dpi=96)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def ImSave(input, mean, std, title=None):
    input = input.numpy().transpose((1, 2, 0))
    input = std * input + mean
    plt.savefig(title + '.png')

# TODO: Do we still need this?
def VisualizeModel(model, numImages=6):
    imagesSoFar = 0
    fig = plt.figure()
    
    for i, data in enumerate(datasetLoaders['test']):
        inputs, labels = data
        if torch.cuda.is_available():
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

# TODO: Look deeper into T-SNE
#try: from sklearn.manifold import TSNE; HAS_SK = True
#except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, datasetClasses[s], backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

def ToVar(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)

def SaveCheckpoint(state, datasetPath, checkpointName='checkpoint.pth'):
    filename = os.path.join(datasetPath, checkpointName)
    torch.save(state, filename)

  # save_checkpoint({
  #           'epoch': epoch + 1,
  #           'model': args.model,
  #           'config': args.model_config,
  #           'state_dict': model.state_dict(),
  #           'best_prec1': best_prec1,
  #           'regime': regime
  #       }, is_best, path=save_path)

# Check t-SNE for details
def TrainModelMiniBatch(model, criterion, optimizer, lr_scheduler, datasetPath,
						datasetLoaders, datasetSizes, trainAccuracyArray,
                        testAccuracyArray, lrLogArray, trainErrorArray, testErrorArray, 
                        startingEpoch = 0, num_epochs=25, saveInterval=5):
    since = time.time()

    #vis = visdom.Visdom()

    # Clear training arrays
    del trainAccuracyArray[:]
    del testAccuracyArray[:]
    del lrLogArray[:]
    del trainErrorArray[:]
    del testErrorArray[:]

    # Set initial values
    best_model = model
    best_acc = 0.0
    currentLr = 0

    # Check if saveInterval is meaningful
    if saveInterval < 0:
        saveInterval = 5

    # For given number of epochs
    for epoch in range(startingEpoch, startingEpoch + num_epochs):
        print('Epoch {}/{}'.format(epoch, startingEpoch + num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                lr_scheduler.step()
                currentLr = lr_scheduler.get_lr()
                print('LR: ' + str(currentLr))

                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(datasetLoaders[phase]):


                #vis.images(inputs.numpy(), opts=dict(title='Input images', caption='How random.'), win=2)

                # Wrap them in Variable
                inputs, labels = ToVar(inputs), ToVar(labels)

                # Debug images
                #ImShow(torchvision.utils.make_grid(inputs.data.cpu()), mean=[0.544, 0.544, 0.544], std=[0.056, 0.056, 0.056], title=labels)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward, backward, optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Accuracy
                _, preds = torch.max(outputs.data, 1)

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                if i % 50 == 0:
                #if True == False:
                	# Minibatch specific info
                    print('epoch {} batch {}/{} loss {:.3f}'.format(
                                epoch, i, len(datasetLoaders[phase]), loss.data[0]))
                    pred_y = preds[1].squeeze()

                    # t-SNE visualization : LATER!
                    if False:
                        # Visualization of trained flatten layer (T-SNE)
                        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                        plot_only = 500
                        low_dim_embs = tsne.fit_transform(outputs.data.cpu().numpy()[:plot_only, :])
                        #labels = labels.numpy()[:plot_only]
                        plot_with_labels(low_dim_embs, labels.data.cpu().numpy())

            epoch_loss = running_loss / datasetSizes[phase]
            epoch_acc = running_corrects / datasetSizes[phase]
            
            if phase == 'train':
                trainErrorArray.append(epoch_loss)
                trainAccuracyArray.append(epoch_acc)
                lrLogArray.append(currentLr)

            if phase == 'test':
                testErrorArray.append(epoch_loss)
                testAccuracyArray.append(epoch_acc)
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)


        print() 

        # Save network on each given interval
        if epoch % saveInterval == 0:
            checkpointName = type(model).__name__ + '_' + str(epoch) + 'checkpoint.pth'
            SaveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, datasetPath, checkpointName)
            print('Checkpoint saved: ' + checkpointName)
            #'arch': args.arch,
            #'best_prec1': best_prec1,

        del inputs, labels, loss, outputs
        gc.collect()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    SaveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': best_model    .state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, datasetPath, type(model).__name__ + '_bestcheckpoint.pth')
    return best_model

# Missing in current PyTorch Windows
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Missing in current PyTorch Windows
class StepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: -0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

# Missing in current PyTorch Windows
class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If True, prints a message to stdout for
            each update. Default: False.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')

# Missing in current PyTorch Windows
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

# Missing in current PyTorch Windows
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

# Missing in current PyTorch Windows
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

# Missing in current PyTorch Windows
def selu(input, inplace=False):
    return SELU_THNN.apply(input, inplace)

# Missing in current PyTorch Windows
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

def PlotActivationMaps(gradients):
    for iT in range(gradients.size()[0]):
        plt.figure(figsize=(12, 12))
        for z in range(gradients.size()[1]):
            plt.subplot(8, 8, z+1)
            plt.title('Level ' + str(iT) + ' field:' + str(z))
            plt.axis('off')
            plt.pause(0.001)
            plt.imshow(gradients[iT, z, :, :].data.cpu().numpy(), interpolation='nearest', cmap='gray')
            
def PlotArrays(arrays, labels, xlabel, ylabel, title):
    p = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel)
    length = len(arrays[0])
    palette = Spectral7[0:len(arrays)]
    x = np.linspace(0, length - 1, length)
    i = 0
    
    for array, label in zip(arrays, labels):
        p.circle(x, array, legend=label, fill_color=palette[i], line_color=palette[i])
        p.line(x, array, legend=label, line_color=palette[i], line_width=2)
        i += 1
    
    p.legend.location = 'bottom_left'
    show(p)

def plotNNFilter(units):
    print(units.shape)
    filters = units.shape[0]
    plt.figure(1, figsize=(6,6))
    
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        #plt.title('Filter ' + str(i))
        plt.axis('off')
        plt.imshow(units[i,0,:,:], interpolation="nearest", cmap="gray")
        
def DetermineAccuracy(net, phase, datasetLoaders):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    correct = 0
    total = 0

    for data in datasetLoaders[phase]:
        # Inputs
        inputs, labels = data
        inputs, labels = ToVar(inputs), ToVar(labels)

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        # TP, FP, TN, FN
        for iResult in range(len(predicted)):
            # TP or TN
            if predicted[iResult].cpu().numpy() == labels.data[iResult]:
                # TP (defect)
                if labels.data[iResult] == 1:
                    tp += 1
                # TN (non-defect)
                else:
                    tn += 1
            # FP or FN
            else:
                # FP
                if labels.data[iResult] == 0:
                    fp += 1
                # TN
                else:
                    fn += 1

        correct += (predicted == labels.data).sum()

    # Sanity
    assert total == (tp + tn + fp + fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    # Also known as sensitivity
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return correct, total, [accuracy, precision, recall, specificity]

def CalculateConfusion(net, datasetClasses, testLoader):
    datasetLen = len(datasetClasses)
    classCorrect = list(0. for i in range(datasetLen))
    classTotal = list(0. for i in range(datasetLen))
    confusion = torch.zeros(datasetLen, datasetLen)

    for i, data in enumerate(testLoader):
        inputs, labels = data
        inputs, labels = ToVar(inputs), ToVar(labels)
    
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        classTotal[labels.data.cpu().numpy()[0]] += 1
        confusion[labels.data.cpu().numpy()[0]][predicted[0][0]] += 1
    
        if labels.data.cpu().numpy()[0] == predicted.cpu().numpy()[0][0]:
            classCorrect[labels.data.cpu().numpy()[0]] += 1
        else: # Display failure cases
            out = torchvision.utils.make_grid(inputs.data.cpu())
            #ImShow(out, mean=0.5, std=0.5, title=datasetClasses[labels.data.cpu()[0]])
            #ImSave(out, 0.5, 0.5, str(i) + str(datasetClasses[labels.data.cpu()[0]]))

    for i, cls in enumerate(classCorrect):
        print('Class ' + datasetClasses[i] + ' total: ' + str(classTotal[i]) + ' correct: ' + str(classCorrect[i]) + ' success rate is ' + str(100 * classCorrect[i] / classTotal[i])) 

    # Normalize confusion matrix
    for i in range(datasetLen):
        confusion[i] = confusion[i] / confusion[i].sum()

    return confusion

class TXTLogger:
    def __init__(self, fileName):
        self.f = open(fileName, 'a')
    # Writes log message into open file
    def Log(self, message):
        self.f.write(str(datetime.datetime.now()) + ': ' + message + '\n')
        self.f.flush()

def EvaluateInference(net, testLoader, testCount=10):
    times = np.zeros(testCount)
    for i in range(times.size):
        input, label = next(iter(testLoader))
        input, label = ToVar(input), ToVar(label)

        startT = time.perf_counter()
        outputs = net(input)
        endT = time.perf_counter()
        
        timems = (endT-startT) * 1000
        times[i] = timems
        print(str(timems) + ' ms')
    return (str(np.mean(times))  + ' ms')
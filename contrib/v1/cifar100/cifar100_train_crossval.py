
# coding: utf-8

# ### This code extends the functionality of https://github.com/pytorch/examples/tree/master/imagenet to support cross-validation training, allowing you compute the out of sample predicted probabilities for the entire imagenet training set: a necessary step for confident learning and the cleanlab package.
# 
# Here is an example of how to use this file:
# 
# ```bash
# # Four fold cross-validation training.
# $ python3 imagenet_train_crossval.py -a resnet18 -b 256 --lr 0.1 --gpu 0 --cvn 4 --cv 0 /IMAGENET_PATH
# $ python3 imagenet_train_crossval.py -a resnet18 -b 256 --lr 0.1 --gpu 1 --cvn 4 --cv 1 /IMAGENET_PATH
# $ python3 imagenet_train_crossval.py -a resnet18 -b 256 --lr 0.1 --gpu 2 --cvn 4 --cv 2 /IMAGENET_PATH
# $ python3 imagenet_train_crossval.py -a resnet18 -b 256 --lr 0.1 --gpu 3 --cvn 4 --cv 3 /IMAGENET_PATH
# 
# # Combine the results
# $ python3 imagenet_train_crossval.py -a resnet18 --cvn 4 --combine-folds /IMAGENET_PATH
# ```

# In[ ]:


# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement


# In[1]:


import argparse
import os
import random
import shutil
import time
import warnings
import sys
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold
import copy
import numpy as np


# In[ ]:


num_classes = 100


# In[19]:


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# In[20]:


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--cv-seed', default=0, type=int,
                    help='seed for determining the cv folds. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cv', '--cv-fold', type=int, default = None,
                    metavar='N', help='The fold to holdout')
parser.add_argument('--cvn', '--cv-n-folds', default = 0, type=int,
                    metavar='N', help='The number of folds')
parser.add_argument('-m', '--dir-train-mask', default = None, type=str,
                    metavar='DIR', help='Boolean mask with True for indices to '
                    'train with and false for indices to skip.')
parser.add_argument('--combine-folds',  action='store_true', default = False, 
                    help='Pass this flag and -a arch to combine probs '
                    'from all folds. You must pass -a and -cvn flags as well!')
parser.add_argument('--train-labels', type=str, default = None, 
                    help='DIR of training labels format: json filename2integer')

best_acc1 = 0


# In[6]:


def main(args = parser.parse_args()):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


# In[ ]:


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    use_crossval = args.cvn > 0
    use_mask = args.dir_train_mask is not None
    cv_fold = args.cv
    cv_n_folds = args.cvn
    class_weights = None
    
    if use_crossval and use_mask:
        raise ValueError('Either args.cvn > 0 or dir-train-mask not None, but not both.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            pretrained=True, num_classes=num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()    
    
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # In case you load checkpoint from different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.507 , 0.487, 0.4417],
                [0.267, 0.256, 0.276],
            ),
        ]),
    )
    
    # if training labels are provided use those instead of dataset labels
    if args.train_labels is not None:
        with open(args.train_labels, 'r') as rf:
            train_labels_dict = json.load(rf)
        train_dataset.imgs = [(fn, train_labels_dict[fn]) for fn, _ in train_dataset.imgs]
        train_dataset.samples = train_dataset.imgs

    # If training only on a cross-validated portion & make val_set = train_holdout.
    if use_crossval:
        checkpoint_fn = "model_{}__fold_{}__checkpoint.pth.tar".format(args.arch, cv_fold)
        print('Computing fold indices. This takes 15 seconds.')
        # Prepare labels
        labels = [label for img, label in datasets.ImageFolder(traindir).imgs]
        # Split train into train and holdout for particular cv_fold.
        kf = StratifiedKFold(n_splits = cv_n_folds, shuffle = True, random_state = args.cv_seed)
        cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]
        # Seperate datasets        
        np.random.seed(args.cv_seed)
        holdout_dataset = copy.deepcopy(train_dataset)
        holdout_dataset.imgs = [train_dataset.imgs[i] for i in cv_holdout_idx]
        holdout_dataset.samples = holdout_dataset.imgs
        train_dataset.imgs = [train_dataset.imgs[i] for i in cv_train_idx]
        train_dataset.samples = train_dataset.imgs
        print('Train size:', len(cv_train_idx), len(train_dataset.imgs))
        print('Holdout size:', len(cv_holdout_idx), len(holdout_dataset.imgs))
    else:
        checkpoint_fn = "model_{}__checkpoint.pth.tar".format(args.arch)
        if use_mask:            
            checkpoint_fn = "model_{}__masked__checkpoint.pth.tar".format(args.arch)
            orig_class_counts = np.bincount(
                [lab for img, lab in datasets.ImageFolder(traindir).imgs])
            train_bool_mask = np.load(args.dir_train_mask)
            # Mask labels
            train_dataset.imgs = [img for i, img in enumerate(train_dataset.imgs) if train_bool_mask[i]]
            train_dataset.samples = train_dataset.imgs
            clean_class_counts = np.bincount(
                [lab for img, lab in train_dataset.imgs])
            print('Train size:', len(train_dataset.imgs))
            # Compute class weights to re-weight loss during training
            # Should use the confident joint to estimate the noise matrix then
            # class_weights = 1 / p(s=k, y=k) for each class k.
            # Here we approximate this with a simpler approach
            # class_weights = count(y=k) / count(s=k, y=k)
            class_weights = torch.Tensor(orig_class_counts / clean_class_counts)
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.507 , 0.487, 0.441],
                [0.267, 0.256, 0.276],
            ),
        ]),
    )


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1, model, and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(best_acc1, acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best = is_best,                
                filename = checkpoint_fn,
                cv_fold = cv_fold,
                use_mask = use_mask,
            )
    if use_crossval:
        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        print("=> loading best model_{}__fold_{}_best.pth.tar".format(args.arch, cv_fold))
        checkpoint = torch.load("model_{}__fold_{}_best.pth.tar".format(args.arch, cv_fold))
        model.load_state_dict(checkpoint['state_dict'])
        print("Running forward pass on holdout set of size:", len(holdout_dataset.imgs))
        probs = get_probs(holdout_loader, model, args)
        np.save('model_{}__fold_{}__probs.npy'.format(args.arch, cv_fold), probs)    


# In[7]:


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


# In[69]:


def get_probs(loader, model, args):

    # switch to evaluate mode
    model.eval()
    ntotal = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / ntotal), end = "")
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs.append(model(input))
    
    # Prepare outputs as a single matrix
    probs = np.concatenate([
        torch.nn.functional.softmax(z, dim = 1) if args.gpu is None else 
        torch.nn.functional.softmax(z, dim = 1).cpu().numpy() 
        for z in outputs
    ])
    
    return probs


# In[8]:


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', cv_fold = None, use_mask = False):
    torch.save(state, filename)
    if is_best:
        sm = "__masked" if use_mask else ""
        sf = "__fold_{}".format(cv_fold) if cv_fold is not None else ""
        wfn = 'model_{}{}{}_best.pth.tar'.format(state['arch'], sm, sf)
        shutil.copyfile(filename, wfn)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    0.1 for epoch [0,150)
    0.01 for epoch [150,250)
    0.001 for epoch [250,350)"""
    
    if epoch < int(60. / 200 * args.epochs):
        lr = args.lr
    elif epoch < int(120. / 200 * args.epochs):
        lr = args.lr * 0.2
    elif epoch < int(160. / 200 * args.epochs):
        lr = args.lr * 0.2 * 0.2
    else:
        lr = args.lr * 0.2 * 0.2 * 0.2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    print('epoch:', epoch+1, '| lr:', lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# In[33]:


def combine_folds(args):
    wfn = 'cifar100__train__model_{}__pyx.npy'.format(args.arch)
    print('Make sure you specified the model architecture with flag -a.')
    print('This method will overwrite file: {}'.format(wfn))
    print('Computing fold indices. This takes 15 seconds.')
    # Prepare labels
    labels = [label for img, label in datasets.ImageFolder(os.path.join(args.data, "train/")).imgs]
    # Intialize pyx array (output of trained network)
    pyx = np.empty((len(labels), num_classes))
    
    # Split train into train and holdout for each cv_fold.
    kf = StratifiedKFold(n_splits = args.cvn, shuffle = True, random_state = args.cv_seed)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(range(len(labels)), labels)):
        probs = np.load('model_{}__fold_{}__probs.npy'.format(args.arch, k))
        pyx[cv_holdout_idx] = probs[:,:num_classes]
    print('Writing final predicted probabilities.')
    np.save(wfn, pyx) 
    
    # Compute overall accuracy
    print('Computing Accuracy.', flush=True)
    acc = sum(np.array(labels) == np.argmax(pyx, axis = 1)) / float(len(labels))
    print('Accuracy: {:.25}'.format(acc))


# In[ ]:


if __name__ == '__main__':
    args = parser.parse_args()
    if args.combine_folds:
        combine_folds(args)
    else:
        main(args)


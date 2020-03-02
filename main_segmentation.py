import argparse
import os
import random
import shutil
import time
import warnings

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
# import torchvision.models as models
import models
from models.segmentation._utils import *
import numpy as np
import math
from models.datasets import CocoStuff164k

model_names = sorted(name for name in models.segmentation.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.segmentation.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--alpha', default=None, type=float,
                    help='Alpha hyperparameter for regularization')
parser.add_argument('--bitW', default=1, type=int,
                    help='bitwidth of weights')
parser.add_argument('--bitA', default=32, type=int,
                    help='bitwidth of activations')
parser.add_argument('--bitScale', default=32, type=int,
                    help='bitwidth of activations')
parser.add_argument('--non-lazy', dest='non_lazy', action='store_true',
                    help='Lazy (STE) or non-lazy projection')
parser.add_argument('--freeze-W', dest='freeze_W', action='store_true',
                    help='Freeze weights to fine-tune batch-norm parameters')
parser.add_argument('--learnable-scalings', dest='learnable_scalings', action='store_true',
                    help='Learnable scalings')
parser.add_argument('--mode', default=None, choices=['layerwise', 'channelwise', 'kernelwise'],
                    help='dataset')

best_acc1 = 0
alpha = None
args = None
num_classes = None

def main():
    global alpha, args, num_classes
    args = parser.parse_args()
    alpha = args.alpha
    num_classes = 22 if args.dataset == 'VOC' else 183  # 21 or 182 (including background) + ambiguous

    # Set up
    set_up()

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


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    ambiguous_label = num_classes - 1

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
    print("=> Creating model '{}'".format(args.arch))
    model = models.segmentation.__dict__[args.arch](num_classes=num_classes)

    # Init learnable scalings
    if args.learnable_scalings:
        for l in layers_list(model):
            l.init_scalings()

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

    if args.freeze_W:
        list(map(freeze_weights, layers_list(model.module)))

    # define loss function (criterion) and optimizer
    class_weights = torch.ones(num_classes)
    class_weights[0] = 1                     # Background
    class_weights[ambiguous_label] = 0       # Ambiguous
    criterion = nn.CrossEntropyLoss(class_weights).cuda(args.gpu)

    if args.non_lazy:
        # SGD (lr=0.1, wd=1e-4) is better for real networks
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        if args.learnable_scalings or int(args.alpha) == 0:
            # Adam (lr=1e-3, wd=1e-6) is better for quantized networks  (except with APSQ)
            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            # APSQ works better with SGD
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch: {} acc1: {:0.2f})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    datafolder_VOC = '/data/Datasets/PASCAL_VOC'
    datafolder_SBD = '/data/Datasets/SBD'
    datafolder_COCO = '/data/Datasets/COCO'

    train_transform = ComposeJoint(
        [
            RandomHorizontalFlipJoint(),
            RandomCropJoint(size=224, pad_if_needed=True),
            [transforms.ToTensor(), lambda x:x],
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), lambda x:x],
            [lambda x:x, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long())],
            [lambda x:x, remap_ambiguous(ambiguous_label)]
        ])
    valid_transform = ComposeJoint(
        [
            RandomCropJoint(size=224, pad_if_needed=True),
            [transforms.ToTensor(), lambda x:x],
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), lambda x:x],
            [lambda x:x, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long())],
            [lambda x:x, remap_ambiguous(ambiguous_label)]
        ])

    train_dataset_VOC = datasets.VOCSegmentation(
        datafolder_VOC, '2012', 'train', download=False, transforms=train_transform)
    train_dataset_SBD = datasets.SBDataset(
        datafolder_SBD, 'train_noval', 'segmentation', download=False, transforms=train_transform)
    train_dataset_Coco = CocoStuff164k(
        root=datafolder_COCO, split='train2017', ignore_label=ambiguous_label, mean_bgr=(104.008, 116.669, 122.675),
        augment=True, base_size=None, crop_size=321, scales=[0.5, 0.75, 1.0, 1.25, 1.5], flip=True)
    val_dataset_VOC = datasets.VOCSegmentation(
        datafolder_VOC, '2012', 'val', download=False, transforms=valid_transform)
    val_dataset_COCO = CocoStuff164k(
        root=datafolder_COCO, split='val2017', ignore_label=ambiguous_label, mean_bgr=(104.008, 116.669, 122.675),
        augment=False)

    train_sampler = None

    train_loader_VOC = torch.utils.data.DataLoader(
        train_dataset_VOC, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_SBD = torch.utils.data.DataLoader(
        train_dataset_SBD, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader_COCO = torch.utils.data.DataLoader(
        train_dataset_Coco, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = val_dataset_VOC if args.dataset == 'VOC' else val_dataset_COCO
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        adjust_alpha(epoch, args)

        # train for one epoch
        if args.dataset == 'VOC':
            train(train_loader_VOC, model, criterion, optimizer, epoch, args)
            train(train_loader_SBD, model, criterion, optimizer, epoch, args)
        elif args.dataset == 'COCO':
            train(train_loader_COCO, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print('Best accuracy: {:0.3f}'.format(best_acc1.item()))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    CE_losses = AverageMeter()
    AP_losses = AverageMeter()
    MIOU = AverageMeter()
    layers = layers_list(model.module)

    # switch to train mode
    model.train()

    # Train using final weights
    if args.freeze_W:
        models.quantized_ops.bitW = args.bitW

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)

        batch_size = target.size(0)
        output = output['out'].view(batch_size, num_classes, -1)
        target = target.view(batch_size, -1)

        CE_loss = criterion(output, target)

        # Angle Regularization
        AP_loss = list(map(angle_penalty_loss, layers))
        AP_loss = sum(AP_loss) / len(AP_loss)
        if int(args.alpha) != 0:
            loss = CE_loss + alpha*AP_loss
        else:
            loss = CE_loss

        # measure accuracy and record loss
        acc = compute_miou(output, target)
        CE_losses.update(CE_loss.item(), input.size(0))
        AP_losses.update(AP_loss.data.item(), input.size(0))
        MIOU.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Loss {loss.avg:.4f} + {AP_loss.avg:.4f}\t'
          'MIOU {MIOU.avg:.3f}\t'
          'alpha {alpha:0.2f}\t'
          'lr {lr:0.1e}'.format(
           epoch, batch_time=batch_time,
           loss=CE_losses, AP_loss=AP_losses, MIOU=MIOU, alpha=alpha,
           lr=optimizer.param_groups[0]['lr']))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    CE_losses = AverageMeter()
    AP_losses = AverageMeter()
    MIOU = AverageMeter()
    layers = layers_list(model.module)

    # switch to evaluate mode
    model.eval()

    # Validate using quantized weights
    models.quantized_ops.bitW = args.bitW

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            # Angle Regularization
            AP_loss = list(map(angle_penalty_loss, layers))
            AP_loss = sum(AP_loss)/len(AP_loss)

            batch_size = target.size(0)
            output = output['out'].view(batch_size, num_classes, -1)
            target = target.view(batch_size, -1)

            CE_loss = criterion(output, target)

            # measure accuracy and record loss
            acc = compute_miou(output, target)
            CE_losses.update(CE_loss.item(), input.size(0))
            AP_losses.update(AP_loss.data.item(), input.size(0))
            MIOU.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test:\t\t'
              'Time {batch_time.avg:.3f}\t'
              'Loss {loss.avg:.4f} + {AP_loss.avg:.4f}\t'
              'MIOU {MIOU.avg:.3f}\t'
              'alpha {alpha:0.2f}'.format(
               batch_time=batch_time, loss=CE_losses, AP_loss=AP_losses, MIOU=MIOU, alpha=alpha))

    if args.non_lazy:
        models.quantized_ops.bitW = 32

    return MIOU.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'pretrained/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'pretrained/model_best.pth.tar')


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_alpha(epoch, args):
    global alpha
    # alpha *= 1.3      # exponential
    alpha = args.alpha ** (epoch // 10 + 1)      # multi-step
    alpha = 10000 if alpha > 10000 else alpha


def compute_miou(output, target):
    eps = 1e-6
    with torch.no_grad():
        # Note: The network shouldn't be labeling pixels as ambiguous
        output = output.max(dim=1).indices                          # Batch x Classes x H*W => B x H*W

        miou = 0

        # Background is considered a class, but not ambiguous
        for k in range(num_classes-1):
            intersection = ((output==k) & (target==k)).int().sum()
            union = ((output==k).int() + (target==k).int()).sum() - intersection
            miou += (intersection.float() + eps) / (union + eps)

        miou /= (num_classes-1)

        return miou


# Create list with conv2d
# NOTE: Make sure layers are declared in order in the model
def layers_list(layer):
    layers = []
    for m_name in layer.__dict__['_modules']:
        m = layer.__dict__['_modules'][m_name]
        if (isinstance(m, nn.Sequential)) or (m.__class__.__name__ == 'IntermediateLayerGetter') or \
                (m.__class__.__name__ == 'BasicBlock') or (m.__class__.__name__ == 'Bottleneck'):
            layers += layers_list(m)
        if isinstance(m, models.quantized_ops.QuantizedConv2d) or isinstance(m, models.quantized_ops.QuantizedConv1d):
            layers += [m]

    return layers


def angle_penalty_loss(layer):
    # Angle of the whole layer
    W_r = layer.weight

    # L1 and L2 binarization for learnable scalings
    # W_q = models.quantized_ops.Binarize_W.apply(W_r)
    # W_q = models.quantized_ops.DoReFa_W(W_r, args.bitW, 'layerwise')
    # if args.mode == 'layerwise':
    #     W_q *= layer.scale
    # elif args.mode == 'channelwise' or isinstance(layer, models.quantized_ops.QuantizedConv1d):
    #     W_q *= layer.scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # elif args.mode == 'kernelwise':
    #     W_q *= layer.scale.unsqueeze(-1).unsqueeze(-1)
    # W_q = W_q.detach()
    # angle = torch.norm(W_r - W_q, p=2)/W_r.nelement()

    #region ### Binary weights ###
    if args.bitW == 1:
        if args.mode == 'layerwise':
            # Layer-wise
            W_r = W_r.view(-1)
            sqrt_n = math.sqrt(len(W_r))
            angle = 1 - (torch.norm(W_r, p=1) / (torch.norm(W_r, p=2)*sqrt_n))

        elif args.mode == 'channelwise':
            # Channel-wise
            ones = torch.ones(W_r.shape[0]).cuda()
            norm_1 = torch.norm(W_r.view(W_r.shape[0], -1), p=1, dim=1)
            norm_2 = torch.norm(W_r.view(W_r.shape[0], -1), p=2, dim=1)
            sqrt_n = math.sqrt(W_r.view(W_r.shape[0], -1).shape[1])
            angle = torch.mean(ones - norm_1 / (norm_2 * sqrt_n))

        elif args.mode == 'kernelwise':
            # Kernel-wise
            if isinstance(layer, models.quantized_ops.QuantizedConv1d):
                ones = torch.ones(W_r.shape[0]).cuda()
                norm_1 = torch.norm(W_r.view(W_r.shape[0], -1), p=1, dim=1)
                norm_2 = torch.norm(W_r.view(W_r.shape[0], -1), p=2, dim=1)
                sqrt_n = math.sqrt(W_r.view(W_r.shape[0], -1).shape[1])
                angle = torch.mean(ones - norm_1 / (norm_2 * sqrt_n))
            else:
                ones = torch.ones(W_r.shape[0], W_r.shape[1]).cuda()
                norm_1 = torch.norm(W_r.view(W_r.shape[0], W_r.shape[1], -1), p=1, dim=2)
                norm_2 = torch.norm(W_r.view(W_r.shape[0], W_r.shape[1], -1), p=2, dim=2)
                sqrt_n = math.sqrt(W_r.view(W_r.shape[0], W_r.shape[1], -1).shape[2])
                angle = torch.mean(ones - norm_1 / (norm_2 * sqrt_n))

    #endregion

    #region ### Quantized k-bit weights ###
    elif args.bitW > 1:
        eps = 1e-8
        W_q = models.quantized_ops.DoReFa_W(W_r, args.bitW, args.mode).detach()
        if args.mode == 'layerwise':
            # Layer-wise
            W_r = W_r.view(-1)
            W_q = W_q.view(-1)
            dot_prod = torch.dot(W_r, W_q)
            norms = torch.norm(W_r, p=2) * torch.norm(W_q, p=2) + eps
            angle = 1 - dot_prod / norms

        elif args.mode == 'channelwise':
            # Channel-wise
            ones = torch.ones(W_r.shape[0]).cuda()
            W_r = W_r.view(W_r.shape[0], -1)
            W_q = W_q.view(W_q.shape[0], -1)
            dot_prod = torch.bmm(W_r.unsqueeze(1), W_q.unsqueeze(-1)).squeeze()
            norms = torch.norm(W_r, p=2, dim=1) * torch.norm(W_q, p=2, dim=1) + eps
            angle = torch.mean(ones - dot_prod / norms)

        elif args.mode == 'kernelwise':
            # Kernel-wise
            ones = torch.ones(W_r.shape[0] * W_r.shape[1]).cuda()
            W_r = W_r.view(-1, W_r.shape[2] * W_r.shape[3])
            W_q = W_q.view(-1, W_q.shape[2] * W_q.shape[3])
            dot_prod = torch.bmm(W_r.unsqueeze(1), W_q.unsqueeze(-1)).squeeze()
            norms = torch.norm(W_r, p=2, dim=1) * torch.norm(W_q, p=2, dim=1) + eps
            angle = torch.mean(ones - dot_prod / norms)

    #endregion

    return angle


def freeze_weights(layer):
    for param in layer.parameters():
        param.requires_grad = False


def set_up():
    # assert not (args.learnable_scalings and args.alpha), "Error: Learning scalings on and alpha != 0"
    assert not (args.alpha and args.freeze_W), "Freeze_W and alpha != 0"
    assert not (args.learnable_scalings and args.non_lazy), "Learnable scalings and non-lazy"
    assert not (args.learnable_scalings and args.freeze_W), "Learnable scalings and freeze-W"

    if args.learnable_scalings:
        print("Warning: Learnable scalings uses Adam, use approapiate lr and wd")

    models.quantized_ops.learnable_scalings = args.learnable_scalings
    models.quantized_ops.mode = args.mode

    if args.non_lazy:
        models.quantized_ops.bitW = 32
    else:
        models.quantized_ops.bitW = args.bitW

    models.quantized_ops.bitA = args.bitA
    models.quantized_ops.bitScale = args.bitScale

if __name__ == '__main__':
    main()

import argparse
import os
import numpy as np
import cv2
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
import torchvision.models as models
from tensorboardX import SummaryWriter

import models_seg
from data_portrait import SegDataset

from losses import dice_loss
from utils import get_vis

from albumentations import (
    Compose,
    HorizontalFlip,
    Resize,
    RandomBrightnessContrast,
    RandomGamma,
    Cutout,
    ShiftScaleRotate
)


model_names = [
        "densenet121", "densenet169", "densenet201", "densenet161",
        "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107",
        ]


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data_train', metavar='DIR',
                    help='path to dataset')
parser.add_argument('data_val', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet161',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: densenet161)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--need-vis', dest='need_vis', action='store_true',
                    help='vis')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='debug')
parser.add_argument('--freeze-bn', dest='freeze_bn', action='store_true',
                    help='freeze-bn')
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

best_acc1 = 0
writer = SummaryWriter(flush_secs=10)

def main():
    args = parser.parse_args()
    args.num_classes = 2

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
    if 'dpn' in args.arch:
        model = models_seg.ACFDPN(num_classes=args.num_classes, backbone=args.arch)
    elif 'dpn' in args.arch:
        model = models_seg.ACFDenseNet(num_classes=args.num_classes, backbone=args.arch)
    else:
        raise Exception("wrong arch")

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
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print("Convering to sync")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
            model = torch.nn.DataParallel(model)
            model = model.cuda()

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = dice_loss

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
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    albu_t = Compose([
            Resize(800, 600),
            HorizontalFlip(),
            Cutout(num_holes=8, max_h_size=800//10, max_w_size=800//10),
            ShiftScaleRotate(),
            RandomBrightnessContrast(),
            RandomGamma(),
        ])

    train_dataset = SegDataset(args.data_train,
        transforms.Compose([
            transforms.Lambda(lambda x: {"image": x[0], "mask": x[1]}),
            transforms.Lambda(lambda x: albu_t(**x)),
            transforms.Lambda(lambda x: (transforms.functional.to_tensor(x['image']), transforms.functional.to_tensor(x['mask']))),
            transforms.Lambda(lambda x: (normalize(x[0]), x[1][0, :, :].long())),
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    albu_t_val = Compose([
            Resize(800, 600),
        ])
    val_loader = torch.utils.data.DataLoader(
        SegDataset(args.data_val,
            transforms.Compose([
                transforms.Lambda(lambda x: {"image": x[0], "mask": x[1]}),
                transforms.Lambda(lambda x: albu_t_val(**x)),
                transforms.Lambda(lambda x: (transforms.functional.to_tensor(x['image']), transforms.functional.to_tensor(x['mask']))),
                transforms.Lambda(lambda x: (normalize(x[0]), x[1][0, :, :].long())),
            ])
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 'e', args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

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
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    dice = AverageMeter('Dice', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, dice],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.freeze_bn:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.debug and i > 10: break
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output_coarse, output = model(images)
        loss = 0.6*criterion(output, target) + 0.7*criterion(output_coarse, target)

        # measure accuracy and record loss
        acc1 = accuracy(output.permute(0, 2, 3, 1).reshape(-1, args.num_classes), target.view(-1), topk=(1,))
        losses.update(loss.item(), output.size(0)*output.size(2)*output.size(3))
        top1.update(acc1[0][0].item(), output.size(0)*output.size(2)*output.size(3))
        dice.update(1 - dice_loss(output, target).item(), output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    writer.add_scalar('acc1/acc1_train', top1.avg, epoch)
    writer.add_scalar('dice/dice_train', dice.avg, epoch)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    dice = AverageMeter('Dice', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, dice],
        prefix='Test: ')


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.debug and i > 10: break
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)


            # compute output
            coarse_output, output = model(images)
            loss = criterion(output, target)

            if args.need_vis:
                for t, (in_dbg, out_dbg, out_gt, in_dbg1) in get_vis(images, output, target):
                    cv2.imwrite("./tmp/{}_{}_targ.png".format(i, t), out_gt)
                    cv2.imwrite("./tmp/{}_{}_in.png".format(i, t), cv2.cvtColor(in_dbg, cv2.COLOR_BGR2RGB))
                    cv2.imwrite("./tmp/{}_{}_out.png".format(i, t), out_dbg)
                    cv2.imwrite("./tmp/{}_{}_crop.png".format(i, t), cv2.cvtColor(in_dbg1, cv2.COLOR_BGRA2RGBA))

            # measure accuracy and record loss
            acc1 = accuracy(output.permute(0, 2, 3, 1).reshape(-1, args.num_classes), target.view(-1), topk=(1,))
            losses.update(loss.item(), output.size(0)*output.size(2)*output.size(3))
            top1.update(acc1[0][0].item(), output.size(0)*output.size(2)*output.size(3))
            dice.update(1 - dice_loss(output, target).item(), output.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Dice {dice.avg:.3f}'
              .format(top1=top1, dice=dice))

    if epoch != 'e':
        writer.add_scalar('acc1/acc1_val', top1.avg, epoch)
        writer.add_scalar('dice/dice_val', dice.avg, epoch)

        _, (in_dbg, out_dbg, out_gt, in_dbg1) = get_vis(images, output, target)[0]
        writer.add_image("a_input", in_dbg.transpose(2,0,1), epoch)
        writer.add_image("b_predicted", out_dbg[..., None].transpose(2,0,1), epoch)
        writer.add_image("c_ground truth", out_gt[..., None].transpose(2,0,1), epoch)
        writer.add_image("d_cropped", in_dbg1.transpose(2,0,1), epoch)

    return dice.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()

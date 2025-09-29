import argparse
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import scd.builder
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using DP/DDP')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')

parser.add_argument('--checkpoint-path', default='./checkpoints/pretrain/', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation for self supervised training (joint or motion or bone)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='training protocol cross_view/cross_subject/cross_setup')

# encoder / MoCo
parser.add_argument('--encoder-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--encoder-k', default=16384, type=int, help='queue size')
parser.add_argument('--encoder-m', default=0.999, type=float, help='momentum for key encoder')
parser.add_argument('--encoder-t', default=0.07, type=float, help='softmax temperature')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use (single GPU)')

# ===== Physics-aware (新增) =====
parser.add_argument('--use-kin-residual', action='store_true',
                    help='enable residual injection: Y <- Y + beta * W(kin_stats)')
parser.add_argument('--use-phys-bias', action='store_true',
                    help='(placeholder) physics-aware bias on attention logits (not used in this encoder)')
parser.add_argument('--use-kv-gate', action='store_true',
                    help='enable speed-based gating on token features (approx K/V gating)')

parser.add_argument('--beta-s', type=float, default=0.0, help='init beta for spatial residual')
parser.add_argument('--beta-t', type=float, default=0.0, help='init beta for temporal residual')
parser.add_argument('--beta-s-max', type=float, default=0.3, help='target beta_s after warm-up')
parser.add_argument('--beta-t-max', type=float, default=0.3, help='target beta_t after warm-up')

parser.add_argument('--lambda-phys', type=float, default=0.0, help='init lambda for physics bias (placeholder)')
parser.add_argument('--lambda-phys-max', type=float, default=0.5, help='target lambda after warm-up (placeholder)')

parser.add_argument('--alpha-gate', type=float, default=0.0, help='init alpha for gating scale')
parser.add_argument('--alpha-gate-max', type=float, default=0.3, help='target alpha after warm-up')
parser.add_argument('--gamma-gate', type=float, default=None, help='gamma for V gating; default=None -> gamma=alpha')

parser.add_argument('--phys-use-acc', action='store_true',
                    help='use acceleration magnitude (|a|) besides |v| for speed score')
parser.add_argument('--phys-mix-a', type=float, default=0.3,
                    help='mixing weight for |a| when computing speed score: s=|v| + phys_mix_a*|a|')

parser.add_argument('--warmup-epochs-phys', type=int, default=15, help='epochs to warm up phys params to *-max')
parser.add_argument('--attn-bias-rownorm', action='store_true',
                    help='row-normalize B before adding to logits (placeholder)')

# ★ 新增：控制“只在 Q 或 K 分支”启用
parser.add_argument('--phys-side-residual', type=str, default='both', choices=['both', 'q', 'k', 'none'],
                    help='apply residual injection on which side (default: both)')
parser.add_argument('--phys-side-gate', type=str, default='both', choices=['both', 'q', 'k', 'none'],
                    help='apply gating on which side (default: both)')

# optional: control MoCo enqueue
parser.add_argument('--no-enqueue', action='store_true', help='disable MoCo queue enqueue/dequeue')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Deterministic CUDNN is enabled; may slow training.')

    # dataset/protocol
    from options import options_pretraining as options
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_part1' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part1_cross_subject()
    elif args.pre_dataset == 'pku_part2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part2_cross_subject()
    else:
        raise ValueError(f'Unsupported dataset/protocol: {args.pre_dataset}/{args.protocol}')

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # model
    print("=> creating model")
    model = scd.builder.SCD_Net(opts.encoder_args, args.encoder_dim, args.encoder_k, args.encoder_m, args.encoder_t)
    print("options", opts.train_feeder_args)
    print(model)

    # single GPU
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # enqueue switch
    if hasattr(model, 'set_enqueue_enabled'):
        model.set_enqueue_enabled(not args.no_enqueue)

    # pass physics config
    phys_cfg = dict(
        use_kin_residual=args.use_kin_residual,
        use_phys_bias=args.use_phys_bias,
        use_kv_gate=args.use_kv_gate,
        phys_use_acc=args.phys_use_acc,
        phys_mix_a=args.phys_mix_a,
        attn_bias_rownorm=args.attn_bias_rownorm
    )
    if hasattr(model, 'set_phys_cfg'):
        model.set_phys_cfg(phys_cfg)

    # side control (★ 新增)
    if hasattr(model, 'set_phys_side'):
        model.set_phys_side(residual_side=args.phys_side_residual,
                            gate_side=args.phys_side_gate)

    # initial strengths
    gamma_gate = args.gamma_gate if args.gamma_gate is not None else args.alpha_gate
    if hasattr(model, 'set_phys_strength'):
        model.set_phys_strength(
            beta_s=args.beta_s, beta_t=args.beta_t,
            lambda_phys=args.lambda_phys,
            alpha_gate=args.alpha_gate, gamma_gate=gamma_gate
        )

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', 'NA')})")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # loader
    train_dataset = get_pretraining_set(opts)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    os.makedirs(args.checkpoint_path, exist_ok=True)
    writer = SummaryWriter(args.checkpoint_path)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # warm-up strengths
        wu = max(1, args.warmup_epochs_phys)
        warm_ratio = min(1.0, (epoch + 1) / wu)
        beta_s = args.beta_s + (args.beta_s_max - args.beta_s) * warm_ratio
        beta_t = args.beta_t + (args.beta_t_max - args.beta_t) * warm_ratio
        lambda_phys = args.lambda_phys + (args.lambda_phys_max - args.lambda_phys) * warm_ratio
        alpha_gate = args.alpha_gate + (args.alpha_gate_max - args.alpha_gate) * warm_ratio
        gamma_gate = alpha_gate if args.gamma_gate is None else args.gamma_gate

        if hasattr(model, 'set_phys_strength'):
            model.set_phys_strength(
                beta_s=beta_s, beta_t=beta_t,
                lambda_phys=lambda_phys,
                alpha_gate=alpha_gate, gamma_gate=gamma_gate
            )

        # logs
        writer.add_scalar('phys/beta_s', beta_s, epoch)
        writer.add_scalar('phys/beta_t', beta_t, epoch)
        writer.add_scalar('phys/lambda', lambda_phys, epoch)
        writer.add_scalar('phys/alpha_gate', alpha_gate, epoch)
        writer.add_scalar('phys/gamma_gate', gamma_gate, epoch)

        # train
        loss, acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        writer.add_scalar('train_loss', loss.avg, global_step=epoch)
        writer.add_scalar('acc', acc1.avg, global_step=epoch)

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.checkpoint_path, f'checkpoint_{epoch:03d}.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()
    end = time.time()

    for i, (q_input, k_input) in enumerate(train_loader):
        data_time.update(time.time() - end)

        q_input = q_input.float().cuda(args.gpu, non_blocking=True)
        k_input = k_input.float().cuda(args.gpu, non_blocking=True)

        (logits_ti, logits_si, logits_it, logits_is,
         labels_ti, labels_si, labels_it, labels_is) = model(q_input, k_input)

        batch_size = logits_si.size(0)

        loss = (criterion(logits_ti, labels_ti.cuda(args.gpu)) +
                criterion(logits_si, labels_si.cuda(args.gpu)) +
                criterion(logits_it, labels_it.cuda(args.gpu)) +
                criterion(logits_is, labels_is.cuda(args.gpu)))

        losses.update(loss.item(), batch_size)

        acc1, _ = accuracy(logits_si, labels_si.cuda(args.gpu), topk=(1, 5))
        top1.update(acc1[0], batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses, top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))


class AverageMeter(object):
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
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

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
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

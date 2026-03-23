import sys
import time
import logging
import random
import numpy as np
import math
from collections import defaultdict

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda import amp
from model.hgnn_v2 import AdaHGNN
from loss import BCELoss, BCELossWithPseudo
from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012, ComputeAccuracy
from utils.checkpoint import load_pretrained_model, save_checkpoint
from config import arg_parse, logger, show_args

global bestPrec
bestPrec = 0

def main():
    global bestPrec

    # random seed
    seed = 42
    # 设置CPU的随机种子
    torch.manual_seed(seed)
    # 设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # 设置python的随机种子
    random.seed(seed)
    # 设置numpy的随机种子
    np.random.seed(seed)

    # Argument Parse
    args = arg_parse()
    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)
    # Create dataloder
    logger.info("==> Creating dataloader...")
    train_loader, test_loader = get_data_loader(args)
    # print(train_loader.dataset.coco.cats)
    # train_label_num = train_loader.dataset.labels.sum(axis=0)
    # test_label_num = test_loader.dataset.labels.sum(axis=0)
    #
    # labels_num = train_label_num + test_label_num
    # print(labels_num)

    # Load the network
    logger.info("==> Loading the network...")

    GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.labels)
    model = AdaHGNN(image_feature_dim=2048,
                       output_dim=2048,
                       word_features=WordFile,
                       args=args)
    model.cuda()
    if args.pretrain_model != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)

    criterion = {
        'BCELoss': BCELoss(reduce=True, size_average=True).cuda()
    }
    for p in model.resnet_101.parameters():
        p.requires_grad = False
    for p in model.resnet_101.layer4.parameters():
        p.requires_grad = True
    for p in model.resnet_101.layer3.parameters():
        p.requires_grad = True
    # for p in model.resnet_101.layer2.parameters():
    #     p.requires_grad=True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999))
    # 学习率调整策略, min代表监控指标停止下降时触发学习率调整, factor是学习率系数, patience代表损失多少个epoch没下降就调整学习率, verbose打印调整信息
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',      # 监控 mAP
    factor=0.1,
    patience=1,
    verbose=True)

    if args.resume != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resume, map_location='cpu')
        bestPrec, args.start_epoch = checkpoint['best_mAP'], checkpoint['epoch']
        print(checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(args.start_epoch, bestPrec))

    model.cuda()
    logger.info("==> Done!\n")

    if args.evaluate:
        Validate(test_loader, model, criterion, 0, args)
        return

    logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024.0 ** 3))

    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_amp = True if "cuda" in device.type else False
    scaler = amp.GradScaler(enabled=enable_amp)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_loss, refinement_monitor_stats = Train(train_loader, model, criterion, optimizer, scaler, scheduler, writer, epoch, args)
        mAP, top1_acc, top3_acc, top5_acc = Validate(test_loader, model, criterion, epoch, args, scheduler)
        scheduler.step(mAP)

        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('top1_acc', top1_acc, epoch)
        writer.add_scalar('top3_acc', top3_acc, epoch)
        writer.add_scalar('top5_acc', top5_acc, epoch)
        for key, value in refinement_monitor_stats.items():
            writer.add_scalar(f'refine/{key}', value, epoch)

        if refinement_monitor_stats:
            logger.info(
                '[Refine] [Epoch {0}]: beta {1:.4f} delta_h_mean_abs {2:.4f} '
                'clamp_upper_ratio {3:.4f} clamp_lower_ratio {4:.4f} clamp_ratio {5:.4f} '
                'h_ref_shift_ratio {6:.4f} sample_var {7:.4f}'.format(
                    epoch,
                    refinement_monitor_stats.get('beta', 0.0),
                    refinement_monitor_stats.get('delta_h_mean_abs', 0.0),
                    refinement_monitor_stats.get('clamp_upper_ratio', 0.0),
                    refinement_monitor_stats.get('clamp_lower_ratio', 0.0),
                    refinement_monitor_stats.get('clamp_ratio', 0.0),
                    refinement_monitor_stats.get('h_ref_shift_ratio', 0.0),
                    refinement_monitor_stats.get('sample_var', 0.0),
                )
            )
        torch.cuda.empty_cache()

        isBest, bestPrec = mAP > bestPrec, max(mAP, bestPrec)
        save_checkpoint(args, {'epoch': epoch, 'state_dict': model.state_dict(), 'best_mAP': mAP}, isBest)

        if isBest:
            logger.info('[Best] [Epoch {0}]: Best mAP is {1:.4f}'.format(epoch, bestPrec))

    writer.close()


def Train(train_loader, model, criterion, optimizer, scaler, scheduler, writer, epoch, args):
    model.train()
    model.resnet_101.eval()
    model.resnet_101.layer3.train()
    model.resnet_101.layer4.train()

    loss = AverageMeter()
    epoch_monitor = defaultdict(AverageMeter)
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()

    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = input.cuda(), target.cuda()
        # Log time of loading data
        data_time.update(time.time() - end)
        # with amp.autocast(enabled=enable_amp):
        # Forward
        output = model(input)
        monitor_stats = model.get_refinement_monitor_stats()
        for key, value in monitor_stats.items():
            epoch_monitor[key].update(value, input.size(0))
        # Compute and log loss
        loss_ = criterion['BCELoss'](output, target)
        # assert torch.isnan(loss_).sum() == 0, print(loss_)
        loss.update(loss_.item(), input.size(0))
        # Backward
        loss_.backward()
        optimizer.step()
        # scaler.scale(loss_).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()
        if batchIndex % args.print_freq == 0:
            logger.info('[Train] [Epoch {0}]: [{1:04d}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                                                        epoch, batchIndex, len(train_loader),
                                                                        batch_time=batch_time, data_time=data_time,
                                                                        loss=loss))
            sys.stdout.flush()


    writer.add_scalar('Loss', loss.avg, epoch)
    epoch_stats = {key: meter.avg for key, meter in epoch_monitor.items()}

    return loss.avg, epoch_stats

def Validate(val_loader, model, criterion, epoch, args,scheduler):

    model.eval()
    apMeter = AveragePrecisionMeter()
    top1_Accuracy = AverageMeter()
    top3_Accuracy = AverageMeter()
    top5_Accuracy = AverageMeter()
    pred, loss, batch_time, data_time = [], AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(val_loader):
        input, target = input.cuda(), target.float().cuda()

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        with torch.no_grad():
            output = model(input)
        # Compute loss and precision
        loss_ = criterion['BCELoss'](output, target)
        loss.update(loss_.item(), input.size(0))
        top_Acc = ComputeAccuracy(output, target)
        top1_acc, top3_acc, top5_acc = top_Acc[0], top_Acc[1], top_Acc[2]
        # print("loss: ", loss_, "\tt1:", top1_acc, "\tt3:", top3_acc, "\tt5:", top5_acc)
        top1_Accuracy.update(top1_acc.item())
        top3_Accuracy.update(top3_acc.item())
        top5_Accuracy.update(top5_acc.item())
        target[target < 0] = 0
        # Compute mAP
        apMeter.add(output, target)
        pred.append(torch.cat((output, (target>0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end =time.time()

        # logger.info information of current batch
        if batchIndex % args.print_freq == 0:
            logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(
                                epoch, batchIndex, len(val_loader),
                                batch_time=batch_time, data_time=data_time, loss=loss))
            sys.stdout.flush()

    pred = torch.cat(pred, 0).cpu().clone().numpy()
    mAP = Compute_mAP_VOC2012(pred, args.classNum)
    averageAP = apMeter.value().mean()
    averageTop1 = top1_Accuracy.avg
    averageTop3 = top3_Accuracy.avg
    averageTop5 = top5_Accuracy.avg
    OP, OR, OF1, CP, CR, CF1 = apMeter.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter.overall_topk(3)

    logger.info('[Test] mAP: {mAP:.4f}, averageAP: {averageAP:.3f}, top1_Acc: {top1_acc:.3f}, top3_Acc: {top3_acc:.3f}, top5_Acc: {top5_acc:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
        mAP=mAP, averageAP=averageAP, top1_acc=averageTop1, top3_acc=averageTop3, top5_acc=averageTop5,
        OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K,
        CF1_K=CF1_K))


    return mAP, averageTop1, averageTop3, averageTop5


if __name__ == '__main__':
    main()

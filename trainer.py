import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, IoULoss
from torchvision import transforms
from lion_pytorch import Lion


def trainer_polyp(args, model, snapshot_path):
    from datasets.dataset_polyp import PolypDataset, get_loader
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations
    trainloader = get_loader(args.img_root, args.gt_root, split="train", batchsize=batch_size, trainsize=args.img_size, augmentation=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    iou_loss = IoULoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0  # not used any where
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            loss_ce = ce_loss(np.squeeze(outputs), np.squeeze(label_batch.long()))
            loss_dice = dice_loss(np.squeeze(outputs), np.squeeze(label_batch.long()), softmax=True)
            loss_iou = iou_loss(np.squeeze(outputs), np.squeeze(label_batch.long()), softmax=True)

            loss = 0.5 * loss_ce + 0.5 * loss_dice
            # loss = 0.33 * loss_ce + 0.33 * loss_dice + 0.33 * loss_iou  # (this is new line of code)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_iou', loss_iou, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            # logging.info('iteration %d : loss : %f, loss_ce: %f loss_iou: %f' % (iter_num, loss.item(), loss_ce.item(), loss_iou.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"





def trainer_cascade(args, model, snapshot_path):
    from datasets.dataset_polyp import PolypDataset, get_loader
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations

    trainloader = get_loader(args.img_root, args.gt_root, split="train", batchsize=batch_size, trainsize=args.img_size, augmentation=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)  # AdamW
    optimizer = Lion(model.parameters(), lr=base_lr, weight_decay=1e-4)  # Lion
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            p1, p2, p3, p4 = model(image_batch) # forward
            
            outputs = p1 + p2 + p3 + p4 # additive output aggregation
            
            loss_ce1 = ce_loss(np.squeeze(p1), np.squeeze(label_batch.long()))
            loss_ce2 = ce_loss(np.squeeze(p2), np.squeeze(label_batch.long()))
            loss_ce3 = ce_loss(np.squeeze(p3), np.squeeze(label_batch.long()))
            loss_ce4 = ce_loss(np.squeeze(p4), np.squeeze(label_batch.long()))
            loss_dice1 = dice_loss(np.squeeze(p1), np.squeeze(label_batch), softmax=True)
            loss_dice2 = dice_loss(np.squeeze(p2), np.squeeze(label_batch), softmax=True)
            loss_dice3 = dice_loss(np.squeeze(p3), np.squeeze(label_batch), softmax=True)
            loss_dice4 = dice_loss(np.squeeze(p4), np.squeeze(label_batch), softmax=True)
            
            
            loss_p1 = 0.3 * loss_ce1 + 0.7 * loss_dice1
            loss_p2 = 0.3 * loss_ce2 + 0.7 * loss_dice2
            loss_p3 = 0.3 * loss_ce3 + 0.7 * loss_dice3
            loss_p4 = 0.3 * loss_ce4 + 0.7 * loss_dice4
            
            alpha, beta, gamma, zeta = 1., 1., 1., 1.
            loss = alpha * loss_p1 + beta * loss_p2 + gamma * loss_p3 + zeta * loss_p4 # current setting is for additive aggregation.
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            if iter_num % 20 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                image = image_batch[1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...] * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
     
        
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))

        save_interval = 50
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
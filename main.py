import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math
import random
import numpy as np

from model.TSTLN import ALL_CNN
from utils import calculate_metrics
from Data import TSdata
from predict import predicted
import time

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def train(train_loader,val_loader, model, criterion1, criterion2, optimizer, epoch, args):
    # switch to train mode
    model.train()
    losses = 0
    val_losses = 0
    tra_losses = 0
    diff_losses = 0
    t_total = 0
    v_total = 0

    for i, (ts,labels) in enumerate(train_loader):
        ts, labels= ts.cuda(), labels.cuda()

        pg1, pg2, zg1, zg2 = model.forward_global(ts)
#
        input_g = torch.cat([zg1, zg2], dim=1)
        loss_tran_g = -(criterion1(pg1, zg2).mean() + criterion1(pg2, zg1).mean()) * 0.5

        pl1, pl2, zl1, zl2 = model.forward_local(ts)
        loss_tran_l = -(criterion1(pl1, zl2).mean() + criterion1(pl2, zl1).mean()) * 0.5

        input_l = torch.cat([zl1, zl2], dim=1)

        loss_tran = (loss_tran_g + loss_tran_l)/2
        #loss_tran =  loss_tran_l

        # compute outputtiff and loss
        output = model.classify(input_g,input_l)
        loss_class = criterion2(output,labels)

        loss = loss_tran + loss_class

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss_class.item()
        tra_losses += loss_tran.item()
        #diff_losses += loss_diff.item()
        t_total += ts.size(0)
        #print(position.grad)

    with torch.no_grad():
        model.eval()
        predicted = torch.tensor([])
        labels_all = torch.tensor([])
        for j, (ts, labels) in enumerate(val_loader):
            ts, labels = ts.cuda(), labels.cuda()

            pg1, pg2, zg1, zg2 = model.forward_global(ts)
#
            input_g = torch.cat([zg1, zg2], dim=1)

            pl1, pl2, zl1, zl2 = model.forward_local(ts)

            input_l = torch.cat([zl1, zl2], dim=1)

            # compute outputtiff and loss
            logits = model.classify(input_g, input_l)
            val_loss_class = criterion2(logits,labels)

            val_loss =  val_loss_class

            val_losses += val_loss.item()
            v_total += ts.size(0)

            predicted_tmp = torch.argmax(logits, 1)
            if j == 0:
                predicted = predicted_tmp
                labels_all = labels
            else:
                predicted = torch.cat([predicted, predicted_tmp], dim = 0)
                labels_all = torch.cat([labels_all, labels], dim = 0)

    acc = calculate_metrics(predicted.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

    print('\tEpoch [{:3d}/{:3d}], Train Loss: {:.4f}, {:.4f}, {:.4f}, Val Loss {:.4f}, Accuracy: {}'
          .format(epoch + 1, args.epochs, losses /t_total, tra_losses, diff_losses, val_losses/v_total, acc))
    #print(args.lr)

    return val_losses/v_total

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    acc_lis = []
    loss_lis= []
    time_s = time.time()
    for seed in range(0, 10):
        loss_lis_tmp = []
        p = argparse.ArgumentParser()
        p.add_argument("--epochs", type=int, default=200)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial (base) learning rate', dest='lr')
        p.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum of SGD solver')
        p.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        p.add_argument('--data', type=int, default=40,
                       help='the number of training data.')
        p.add_argument("--kernel", type=int, default=41)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--heads", type=int, default=1)
        args = p.parse_args()

        setup_seed(seed)
        model = ALL_CNN(depth = args.depth, heads = args.heads, kernel_size = args.kernel).cuda()
        TSdata_load = TSdata.TSLoader('C:/code/pycharm/dataset/rs/0512qzwts.csv', args.data / 100,(args.data + 10) / 100, seed)

        train_loader, val_loader, test_loader = TSdata_load.return_data()
        train_dataset = torch.utils.data.DataLoader(
            train_loader, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, drop_last=False)
        val_dataset = torch.utils.data.DataLoader(
            val_loader, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, drop_last=False)
        test_dataset = torch.utils.data.DataLoader(
            test_loader, batch_size=args.batch_size, shuffle=True,
            pin_memory=False, drop_last=False)
        optim_params = model.parameters()
        init_lr = args.lr

        optimizer = torch.optim.Adam(optim_params, init_lr)
        criterion1 = nn.CosineSimilarity(dim=1).cuda()
        criterion2 = nn.CrossEntropyLoss().cuda()
        acc_best = 1

        for epoch in range(0, args.epochs):
            #adjust_learning_rate(optimizer, init_lr, epoch, args)
            # train for one epoch
            acc = train(train_dataset, val_dataset, model, criterion1, criterion2, optimizer, epoch, args)
            loss_lis_tmp.append(acc)
            if acc <= acc_best:
                acc_best = acc
                torch.save(model.state_dict(),"./modelsave/SAqzw.pth")
            print('current best accuarcy',acc_best)

        best_model = ALL_CNN(depth = args.depth, heads = args.heads, kernel_size = args.kernel).cuda()
        best_model.load_state_dict(torch.load('./modelsave/SAqzw.pth'))

        pre_acc = predicted(test_dataset,best_model)
        print(pre_acc)
        acc_lis.append(pre_acc)

    time_e = time.time()
    print(time_e-time_s)
    acc_lis = np.array(acc_lis)
    mean = np.expand_dims(np.mean(acc_lis, axis=0), axis=0)
    std = np.expand_dims(np.std(acc_lis, axis=0), axis=0)
    acc_lis = np.concatenate((acc_lis, mean, std), axis=0)

    print("\033[1;33;44mlength:{:1d}\033[0m".format(1), mean, std)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import math


from utils import calculate_metrics
from Data import TSdata


def predicted(test_loader, model):
    with torch.no_grad():
        model.eval()
        predicted = torch.tensor([])
        labels_all = torch.tensor([])
        for j, (ts, labels) in enumerate(test_loader):
            ts, labels = ts.cuda(), labels.cuda()

            pg1, pg2, zg1, zg2 = model.forward_global(ts)

            input_g = torch.cat([zg1, zg2], dim=1)

            pl1, pl2, zl1, zl2 = model.forward_local(ts)

            input_l = torch.cat([zl1, zl2], dim=1)

            logits = model.classify(input_g, input_l)

            predicted_tmp = torch.argmax(logits, 1)
            if j == 0:
                predicted = predicted_tmp
                labels_all = labels
            else:
                predicted = torch.cat([predicted, predicted_tmp], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)

        acc = calculate_metrics(predicted.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

        return acc

def predicted_notra(test_loader, model):
    with torch.no_grad():
        model.eval()
        predicted = torch.tensor([])
        labels_all = torch.tensor([])
        for j, (ts, labels) in enumerate(test_loader):
            ts, labels = ts.cuda(), labels.cuda()

            zg1, zg2 = model.forward_global(ts)

            input_g = torch.cat([zg1, zg2], dim=1)

            zl1, zl2 = model.forward_local(ts)

            input_l = torch.cat([zl1, zl2], dim=1)

            logits = model.classify(input_g, input_l)

            predicted_tmp = torch.argmax(logits, 1)
            if j == 0:
                predicted = predicted_tmp
                labels_all = labels
            else:
                predicted = torch.cat([predicted, predicted_tmp], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)

        acc = calculate_metrics(predicted.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

        return acc

def predicted_global(test_loader, model):
    with torch.no_grad():
        model.eval()
        predicted = torch.tensor([])
        labels_all = torch.tensor([])
        for j, (ts, labels) in enumerate(test_loader):
            ts, labels = ts.cuda(), labels.cuda()

            pg1, pg2, zg1, zg2 = model.forward_global(ts)

            input_g = torch.cat([zg1, zg2], dim=1)

            logits = model.classify(input_g)

            predicted_tmp = torch.argmax(logits, 1)
            if j == 0:
                predicted = predicted_tmp
                labels_all = labels
            else:
                predicted = torch.cat([predicted, predicted_tmp], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)

        acc = calculate_metrics(predicted.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

        return acc

def predicted_local(test_loader, model):
    with torch.no_grad():
        model.eval()
        predicted = torch.tensor([])
        labels_all = torch.tensor([])
        for j, (ts, labels) in enumerate(test_loader):
            ts, labels = ts.cuda(), labels.cuda()

            pl1, pl2, zl1, zl2 = model.forward_local(ts)

            input_l = torch.cat([zl1, zl2], dim=1)

            logits = model.classify(input_l)

            predicted_tmp = torch.argmax(logits, 1)
            if j == 0:
                predicted = predicted_tmp
                labels_all = labels
            else:
                predicted = torch.cat([predicted, predicted_tmp], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)

        acc = calculate_metrics(predicted.detach().cpu().numpy(), labels_all.detach().cpu().numpy())

        return acc
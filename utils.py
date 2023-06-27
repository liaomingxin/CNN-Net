import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import Sampler, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from Resnet import *
from dataset import *
from model import * 



def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(backbone, pretrain=True, classes_num=200):
    print('==> Building model..')
    feature_size = 512
    if backbone == 'resnet50':
        num_ftrs = 2048
        basenet = resnet50(pretrained=pretrain)
        net = CNN_Net(basenet, feature_size, num_ftrs, classes_num)
    elif backbone == 'resnet101':
        num_ftrs = 2048
        basenet = resnet101(pretrained=pretrain)
        net = CNN_Net(basenet, feature_size, num_ftrs, classes_num)
    elif backbone == 'resnet34':
        num_ftrs = 512
        basenet = resnet34(pretrained=pretrain)
        net = CNN_Net(basenet, feature_size, num_ftrs, classes_num)
    return net

def smooth_CE(logits, label, peak):
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1 - peak) / (num_cls - 1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss
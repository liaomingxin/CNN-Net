import os
import time
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset import *
from utils import *
from state import * 
from test import test

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train():
    parser = argparse.ArgumentParser('FGVC', add_help=False)
    parser.add_argument('--tag', type=str, required=True, help="exp tag")
    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size for training")
    parser.add_argument('--resume', type=str, default="", help="resume from saved model path")
    parser.add_argument('--dataset_name', type=str, default="cub", help="dataset name")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--wkl', type=float, default=2.0, help="weight of KL-Loss")
    args, _ = parser.parse_known_args()

    epochs = args.epochs
    batch_size = args.batch_size

    ## Data
    data_config = { "数据集1": [200, "/home/数据集1/"] }
    dataset_name = args.dataset_name
    classes_num, data_root = data_config[dataset_name]
    if dataset_name == 'air':
        trainset = Signature1(root=data_root, is_train=True, data_len=None)
        testset = Signature1(root=data_root, is_train=False, data_len=None)
    num_workers = 8 if torch.cuda.is_available() else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    exp_dir =  'output/' + time_str + '_' + dataset_name + '_' + args.backbone + '_' + str("test")
    os.makedirs(exp_dir, exist_ok=True)

    # log config
    msg = ''   # log string
    msg += '\n----------------- Configs ---------------\n'
    for (k, v) in vars(args).items():
        msg += '{:>25}: {:<30}\n'.format(str(k), str(v))   # format(key, value)
    msg += '----------------- End -------------------\n'
    print(msg)
    with open(exp_dir + '/results_train.txt', 'a') as file:   # log file
        file.write(msg)
    
    ## Model
    if args.resume != "":
        net = torch.load(args.resume)
    else:
        net = load_model(backbone=args.backbone, pretrain=True, classes_num=classes_num)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.to(device)
        netp = torch.nn.DataParallel(net)
    else:
        device = torch.device('cpu')
        netp = net

    ## Train
    # CELoss = nn.CrossEntropyLoss()
    finetune_layer_params, fresh_layer_params = net.get_params()
    optimizer = optim.SGD(
        [{'params': fresh_layer_params,    'lr': args.lr}, 
         {'params': finetune_layer_params, 'lr': args.lr / 10.0}],
         lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    start = time.time()
    max_val_acc = 0
    for epoch in range(1, epochs + 1):
        print('\nEpoch: %d' % epoch)

        # update learning rate
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr / 10.0)
        result_str = 'Iteration %d (train) | LR: %.6f' % (epoch, optimizer.param_groups[0]['lr'])
        print(result_str)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(result_str)

        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
        losses4 = AverageMeter()
        losses_all = AverageMeter()
        scores1 = AverageAccMeter()
        scores2 = AverageAccMeter()
        scores3 = AverageAccMeter()
        scores4 = AverageAccMeter()
        scores_all = AverageAccMeter()

        net.train()
        pbar = tqdm(trainloader, dynamic_ncols=True, total=len(trainloader))
        for _, (inputs, targets) in enumerate(pbar):
            if inputs.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
            # forward
            optimizer.zero_grad()
            y1, y2, y3, y4, _, _, _ = netp(inputs)

            scores1.add(y1.data, targets)
            scores2.add(y2.data, targets)
            scores3.add(y3.data, targets)
            scores4.add(y4.data, targets)
            scores_all.add((y1 + y2 + y3 + y4).data, targets)

            loss1 = smooth_CE(y1, targets, 1) * 0.8
            loss2 = smooth_CE(y2, targets, 1) * 0.9
            loss3 = smooth_CE(y3, targets, 1) * 1.0
            loss4 = smooth_CE(y4, targets, 1) * 1.1
            loss = loss1 + loss2 + loss3 + loss4

            losses1.add(loss1.item(), inputs.size(0))
            losses2.add(loss2.item(), inputs.size(0))
            losses3.add(loss3.item(), inputs.size(0))
            losses4.add(loss4.item(), inputs.size(0))
            losses_all.add(loss.item(), inputs.size(0))
   
            # backward
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=losses_all.value(), score=scores_all.value())

        ## result
        result_str = 'Iteration %d (train) | loss1 = %.5f | loss2 = %.5f | loss3 = %.5f | loss4 = %.5f | loss_all = %.5f | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f \n' % (
            epoch, losses1.value(), losses2.value(), losses3.value(), losses4.value(), losses_all.value(), scores1.value(), scores2.value(), scores3.value(), scores4.value())
        print(result_str)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(result_str)

        if epoch < 5 or epoch > (epochs // 2):
            with torch.no_grad():
                loss1, loss2, loss3, loss4, loss_all, score1, score2, score3, score4, score_all = test(net, testset, batch_size)
            if score_all > max_val_acc:
                max_val_acc = score_all
                net.cpu()
                torch.save(net.state_dict(), './' + exp_dir + '/model.pth')
                net.to(device)

            result_str = 'Iteration %d | loss1 = %.5f | loss2 = %.5f | loss3 = %.5f | loss4 = %.5f | loss_all = %.5f | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_all = %.5f \n' % (
                epoch, loss1, loss2, loss3, loss4, loss_all, score1, score2, score3, score4, score_all)
            print(result_str)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write(result_str)
    result_str = 'Best val acc: {:.5f}%, cost: {:.5f}h'.format(max_val_acc, (time.time() - start) / 3600)
    print(result_str)
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write(result_str)

if __name__ == "__main__":
    train()

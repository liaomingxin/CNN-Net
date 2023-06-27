import torch
import torch.nn as nn
from tqdm import tqdm

from state import *

CELoss = nn.CrossEntropyLoss()

def test(net, testset, batch_size):
    
    device = torch.device('cuda')
    num_workers = 16 if torch.cuda.is_available() else 0
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size//2, shuffle=False, num_workers=num_workers, drop_last=False)

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    losses4 = AverageMeter()
    losses_logits = AverageMeter()
    losses_kg = AverageMeter()
    losses_all = AverageMeter()
    scores1 = AverageAccMeter()
    scores2 = AverageAccMeter()
    scores3 = AverageAccMeter()
    scores4 = AverageAccMeter()
    scores_logits = AverageAccMeter()
    scores_kg = AverageAccMeter()
    scores_all = AverageAccMeter()

    net.eval()

    pbar = tqdm(testloader, dynamic_ncols=True, total=len(testloader))
    for _, (inputs, targets) in enumerate(pbar):
        if torch.cuda.is_available():
            inputs, targets = inputs.to(device), targets.to(device)
        y1, y2, y3, y4, _, _, _, logits, similarity = net(inputs)

        scores1.add(y1.data, targets)
        scores2.add(y2.data, targets)
        scores3.add(y3.data, targets)
        scores4.add(y4.data, targets)
        scores_logits.add(logits.data, targets)
        scores_kg.add(similarity.data, targets)
        scores_all.add((y1 + y2 + y3 + y4 + logits).data, targets)

        loss1 = CELoss(y1, targets) * 1
        loss2 = CELoss(y2, targets) * 1
        loss3 = CELoss(y3, targets) * 1
        loss4 = CELoss(y4, targets) * 1
        loss_logit = CELoss(logits, targets) * 1
        loss_kg = CELoss(similarity, targets) * 1
        loss = loss1 + loss2 + loss3 + loss4 + loss_logit + loss_kg

        losses1.add(loss1.item(), inputs.size(0))
        losses2.add(loss2.item(), inputs.size(0))
        losses3.add(loss3.item(), inputs.size(0))
        losses4.add(loss4.item(), inputs.size(0))   
        losses_logits.add(loss_logit.item(), inputs.size(0))
        losses_kg.add(loss_kg.item(), inputs.size(0))
        losses_all.add(loss.item(), inputs.size(0))

    return losses1.value(), losses2.value(), losses3.value(), losses4.value(), losses_logits.value(), losses_kg.value(), losses_all.value(), \
            scores1.value(), scores2.value(), scores3.value(), scores4.value(), scores_logits.value(), scores_kg.value(), scores_all.value()
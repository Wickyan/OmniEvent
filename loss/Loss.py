import torch
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        ce_loss = F.cross_entropy(prediction, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def focal_loss_and_accuracy(prediction, target):
    focal_loss = FocalLoss()
    loss = focal_loss(prediction, target)
    # print(loss, prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    # print(loss, prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def cross_entropy_loss_and_accuracy2(prediction, target):
    valid_mask = (target != 0)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = cross_entropy_loss(prediction, target)
    
    if valid_mask.any():
        filtered_pred = prediction[valid_mask]
        filtered_target = target[valid_mask]
        accuracy = (filtered_pred.argmax(1) == filtered_target).float().mean()
    else:
        accuracy = torch.tensor(0.0).to(prediction.device)
    
    return loss, accuracy

def FCN_loss_and_accuracy(prediction, target, f1, f2):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.L1Loss()
    cla_loss = cross_entropy_loss(prediction, target)
    fus_loss = mse_loss(f1, f2)
    # print(cla_loss, fus_loss)
    loss = cla_loss + fus_loss
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def BCE_and_accuracy(prediction, labels):
    target = 1-labels
    target = torch.stack([target, labels],dim=1)
    cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
    loss = cross_entropy_loss(prediction, target.float())
    # print(loss, prediction, target)
    accuracy = (prediction.argmax(1) == target[:,1]).float().mean()
    return loss, accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ["LaneLoss", "LaneLossS"]


class LaneLossS(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.4, 1, 1])
        self.bce_loss = nn.BCELoss()
        self.soft_miou_loss = SoftIoULoss(3)

    def forward(self, outputs, labels):
        seg_maps, exist_codes = outputs["seg_maps"], outputs["exist_codes"]
        seg_lbl, exist_lbl = labels
        focal_loss = self.focal_loss(seg_maps, seg_lbl)
        iou_loss = self.soft_miou_loss(seg_maps, seg_lbl)
        bce_loss = self.bce_loss(exist_codes, exist_lbl)
        loss = focal_loss + iou_loss + 0.1 * bce_loss

        return loss


class LaneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.4, 1, 1, 1, 1])
        self.bce_loss = nn.BCELoss()
        self.soft_miou_loss = SoftIoULoss(5)

    def forward(self, outputs, labels):
        seg_maps, exist_codes = outputs["seg_maps"], outputs["exist_codes"]
        seg_lbl, exist_lbl = labels
        focal_loss = self.focal_loss(seg_maps, seg_lbl)
        iou_loss = self.soft_miou_loss(seg_maps, seg_lbl)
        bce_loss = self.bce_loss(exist_codes, exist_lbl)
        loss = focal_loss +  iou_loss + 0.1 * bce_loss

        return loss


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter_ = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter_.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - inter_
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1-loss.mean()


class OhemCELoss(nn.Module):

    def __init__(self, thresh, n_min):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return F.cross_entropy(output, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, x, gt):
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1, 2)    # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1, x.size(2))   # N,H*W,C => N*H*W,C
        gt = gt.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, gt)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, gt.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

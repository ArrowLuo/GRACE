"""
@ArrowLuo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    one_hot_mask = torch.arange(0, num_classes).long().repeat(batch_size, 1)
    if target.is_cuda:
        one_hot_mask = one_hot_mask.cuda(target.data.get_device())
    one_hot_mask = one_hot_mask.eq(target.data.repeat(num_classes, 1).t())
    return logits.masked_select(one_hot_mask)

def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1

    target_pt = target.view(-1, 1)
    logpt = F.log_softmax(logits, dim=-1)
    logpt = logpt.gather(1, target_pt)
    loss = -logpt.view(-1)

    if weights is not None:
        weights = class_select(weights, target)
        loss = loss * weights
    return loss

class WeightedCrossEntropy(nn.Module):
    """
    Cross entropy with instance-wise weights.
    """
    def __init__(self, aggregate='mean', ignore_index=-100):
        super(WeightedCrossEntropy, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.ignore_index = ignore_index

    def forward(self, input, target, weights=None):
        target_mask = (target != self.ignore_index)
        target_ignored = target.clone()
        target_ignored[target_ignored < 0] = 0.
        target_ignored = target_ignored.to(dtype=torch.int64)
        ce_loss = cross_entropy_with_weights(input, target_ignored, weights)

        # Note: below operation will be error when labels are ignored
        ce_loss_sum = torch.sum(ce_loss * target_mask)
        target_mask_sum = torch.sum(target_mask)
        if self.aggregate == 'sum' or target_mask_sum == 0:
            return ce_loss_sum
        elif self.aggregate == 'mean':
            return ce_loss_sum / target_mask_sum
        elif self.aggregate is None:
            return ce_loss
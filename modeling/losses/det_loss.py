import torch.nn.functional as F
from ..layers import gather_feature
import torch

def reg_l1_loss(output, mask, index, target, norm_wh = False):
    pred = gather_feature(output, index, use_transform=True)
    try:
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    except:
        mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    # print(pred, target, mask)
    norm_ = torch.sum(target, dim=1, keepdim=True) / 2. + 1e-4 if norm_wh else 1.0

    loss = F.l1_loss(pred * mask / norm_, target * mask / norm_, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss

def mse_loss(pred, gt):
    return torch.nn.MSELoss()(pred, gt)

def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    # print(f'num_pos {num_pos},pos_loss {pos_loss},neg_loss {neg_loss}')
    return loss

def ignore_unlabel_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    # modify neg_inds, set unlabeled gt to 0
    neg_inds = gt.lt(1).float()
    N, C, H, W = gt.shape
    for n in range(N):
        for c in range(C):
            if gt[n, c, :, :].sum() == 0.0:
                neg_inds[n, c, :, :] = 0.0

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss
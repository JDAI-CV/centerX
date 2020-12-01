import torch
import torch.nn as nn

class _nms(nn.Module):
    def __init__(self):
        super(_nms, self).__init__()
        kernel = 3
        pad = (kernel - 1) // 2
        self.maxpool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=pad)

    def forward(self, heat):
        hmax = self.maxpool(heat)
        keep = (hmax == heat).float()
        return heat * keep


class extra_conv(nn.Module):
    def __init__(self):
        super(extra_conv, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, _input):
        output = {}
        output['cls'] = self.leakyrelu(_input['cls'])
        output['mask_cls'] = self.relu(_input['cls'])
        output['wh'] = _input['wh']
        output['reg'] = _input['reg']
        return output


class KdLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(KdLoss, self).__init__()
        self.crit = torch.nn.MSELoss()
        self._nms = _nms()
        self.crit_reg = torch.nn.L1Loss(reduction='sum')
        self.crit_wh = torch.nn.L1Loss(reduction='sum')
        self.teacher_extra_conv = extra_conv()
        self.student_extra_conv = extra_conv()
        self.kd_param = cfg.MODEL.CENTERNET.KD

    def forward(self, student_output, teacher_output, idx): # model_ts, images):
        student_output = self.student_extra_conv(student_output)

        teacher_output = self.teacher_extra_conv(teacher_output)

        s_idx = 0 if idx==0 else self.kd_param.KD_CLS_INDEX[idx-1]
        e_index = self.kd_param.KD_CLS_INDEX[idx]
        kd_weight = self.kd_param.KD_WEIGHT[idx]
        kd_cls_weight = kd_weight * self.kd_param.KD_CLS_WEIGHT[idx]
        kd_wh_weight = kd_weight * self.kd_param.KD_WH_WEIGHT[idx]
        kd_reg_weight = kd_weight * self.kd_param.KD_REG_WEIGHT[idx]
        norm_wh = self.kd_param.NORM_WH[idx]

        _wh_mask = self._nms(teacher_output['mask_cls'])
        wh_mask = torch.max(_wh_mask, 1, keepdim=True).values
        mask_weight = wh_mask.sum() + 1e-4
        kd_cls_loss = self.crit(student_output['cls'][:, s_idx:e_index, :, :],
                                teacher_output['cls'])

        norm_ = torch.sum(teacher_output['wh'], dim=1, keepdim=True) / 2. + 1e-4 if norm_wh else 1.0
        #norm_ = 1.0
        kd_wh_loss = (self.crit_wh((student_output['wh'] * wh_mask) / norm_,
                                   (teacher_output['wh'] * wh_mask) / norm_) /
                      mask_weight)
        kd_reg_loss = (self.crit_wh(student_output['reg'] * wh_mask,
                                    teacher_output['reg'] * wh_mask) /
                       mask_weight)

        kd_cls_loss = kd_cls_loss * kd_cls_weight
        kd_wh_loss =  kd_wh_loss * kd_wh_weight
        kd_reg_loss = kd_reg_loss * kd_reg_weight
        kd_loss_stats = {f'kd_cls_loss{idx}': kd_cls_loss,
                         f'kd_wh_loss{idx}': kd_wh_loss,
                         f'kd_off_loss{idx}': kd_reg_loss}
        return kd_loss_stats

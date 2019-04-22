import torch
import torch.nn as nn

import utils

class textboxesLoss(nn.Module):
    def __init__(self, alpha=1):
        super(textboxesLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, out, lines):
        default_boxes = torch.from_numpy(utils.create_defaul_boxes()) # loacl, boxes
        lines = torch.from_numpy()
        out_boxes = out[:, :, 24:] # batch, local, boxes
        batch, _, _ = out_boxes.size()
        default_boxes = default_boxes.unsqueeze(0).repeat((batch, 1, 1))

        x_d_box = default_boxes[:, :, ::4]
        y_d_box = default_boxes[:, :, 1::4]
        w_d_box = default_boxes[:, :, 2::4]
        h_d_box = default_boxes[:, :, 3::4]

        x_box = out_boxes[:, :, ::4]
        y_box = out_boxes[:, :, 1::4]
        w_box = out_boxes[:, :, 2::4]
        h_box = out_boxes[:, :, 3::4]

        x = x_d_box + w_d_box * x_box
        y = y_d_box + h_d_box * y_box
        w = w_d_box * torch.exp(w_box)
        h = h_d_box * torch.exp(h_box)

        lt_x = x - w / 2
        lt_y = y - h / 2
        rd_x = x + w / 2
        rd_y = y + h / 2

        for line in lines:
            line = line.unsqueeze(1).repeat((1, lt_x.size()[0], 1))
            lt_x = lt_x.unsqueeze(0).repeat((line.size()[0], 1, 1))

        




    def compute_iou(self, pre, gt):
        W = min(pre[2], gt[2]) - max(pre[0], gt[0])
        H = min(pre[3], gt[3]) - max(pre[1], gt[1])
        if W <= 0 or H <= 0:
            return 0
        SA = (pre[2]-pre[0]) * (pre[3]-pre[1])
        SB = (gt[2]-gt[0]) * (gt[3]-gt[1])
        cross = W * H
        return cross / (SA+SB-cross)



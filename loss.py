import numpy as np
import torch as t
from numpy import random as rd
from torch import nn


class YOLOLoss(nn.Module):

    def __init__(self, S, B, num_classes, lamda_coord, lamda_noobj):
        super(YOLOLoss, self).__init__()
        self.lamda_coord = lamda_coord
        self.lamda_noobj = lamda_noobj
        self.B = B
        self.S = S
        self.num_classes = num_classes

    def forward(self, model_output, targets, orig_image_sizes):
        """

        :param model_output: output of network, shape like (N, S, S, B * 5 + num_classes)
        :param targets: load from loader, shape like (N, S, S, B * 5 + num_classes)
        :param orig_image_sizes: load from loader, shape like (N, 2), every row means (height, width)
        :return:
        """
        loss_grid_all = 0
        grid_height = orig_image_sizes[:, 0] / self.S  # shape like (N,)
        grid_width = orig_image_sizes[:, 1] / self.S  # shape like (N,)
        sample_indexs, y_grid_indexs, x_grid_indexs, bbox_indexs = t.where(targets[:, :, :, self.B * 4:self.B * 5] == 1)  # find all indexs of confidences of targets whose value equal to 1
        obj_grid_indexs = list(zip(sample_indexs.cpu().numpy(), y_grid_indexs.cpu().numpy(), x_grid_indexs.cpu().numpy()))  # get all grid indexs that has object
        obj_grid_unique_indexs = list(set(obj_grid_indexs))
        noobj_bbox_conf_index = t.sum(targets[:, :, :, self.B * 4:self.B * 5], dim=3) == 0
        noobj_bbox_pred_conf = model_output[noobj_bbox_conf_index][:, self.B * 4:self.B * 5]
        loss_noobj_conf = self.calc_noobj_conf_loss(noobj_bbox_pred_conf)
        for grid_index in obj_grid_unique_indexs:
            current_img_grid_height = grid_height[grid_index[0]]
            current_img_grid_width = grid_width[grid_index[0]]
            orig_image_size = orig_image_sizes[grid_index[0]]
            x_base = grid_index[2] * current_img_grid_width
            y_base = grid_index[1] * current_img_grid_height

            current_grid_obj_count = obj_grid_indexs.count(grid_index)

            current_grid_target_info = targets[grid_index[0], grid_index[1], grid_index[2]]
            current_grid_pred_info = model_output[grid_index[0], grid_index[1], grid_index[2]]

            current_grid_pred_class_info = current_grid_pred_info[-self.num_classes:]
            current_grid_pred_xy_info = current_grid_pred_info[:self.B * 2]
            current_grid_pred_wh_info = current_grid_pred_info[self.B * 2:self.B * 4]
            current_grid_pred_conf_info = current_grid_pred_info[self.B * 4:self.B * 5]

            current_grid_target_class_info = current_grid_target_info[-self.num_classes:]
            has_obj_bbox_target_xy_info = current_grid_target_info[:current_grid_obj_count * 2]  # (x1, y1, x2, y2, ...,x_obj, y_obj)
            has_obj_bbox_target_wh_info = current_grid_target_info[self.B * 2:self.B * 2 + current_grid_obj_count * 2]  # (w1, h1, w2, h2, ...,w_obj, h_obj)
            has_obj_bbox_target_conf_info = current_grid_target_info[self.B * 4:self.B * 4 + current_grid_obj_count]  # (conf_1, conf_2, ...,conf_obj)

            try:
                xy_target_add = has_obj_bbox_target_xy_info * t.tensor([current_img_grid_width, current_img_grid_height] * current_grid_obj_count)
                xy_pred_add = current_grid_pred_xy_info * t.tensor([current_img_grid_width, current_img_grid_height] * self.B)
                current_grid_pred_wh_true = (current_grid_pred_wh_info * t.tensor([orig_image_size[1], orig_image_size[0]] * self.B)).view((self.B, 2))
                current_grid_target_wh_true = has_obj_bbox_target_wh_info * t.tensor([orig_image_size[1], orig_image_size[0]] * current_grid_obj_count).view((current_grid_obj_count, 2))
                current_grid_target_xy_true = xy_target_add.view((current_grid_obj_count, 2)) + t.tensor([x_base, y_base])
                current_grid_pred_xy_true = xy_pred_add.view((self.B, 2)) + t.tensor([x_base, y_base])
            except RuntimeError:
                xy_target_add = has_obj_bbox_target_xy_info * t.tensor([current_img_grid_width, current_img_grid_height] * current_grid_obj_count).cuda(0)
                xy_pred_add = current_grid_pred_xy_info * t.tensor([current_img_grid_width, current_img_grid_height] * self.B).cuda(0)
                current_grid_pred_wh_true = (current_grid_pred_wh_info * t.tensor([orig_image_size[1], orig_image_size[0]] * self.B).cuda(0)).view((self.B, 2))
                current_grid_target_wh_true = (has_obj_bbox_target_wh_info * t.tensor([orig_image_size[1], orig_image_size[0]] * current_grid_obj_count).cuda(0)).view((current_grid_obj_count, 2))
                current_grid_target_xy_true = xy_target_add.view((current_grid_obj_count, 2)) + t.tensor([x_base, y_base]).cuda(0)
                current_grid_pred_xy_true = xy_pred_add.view((self.B, 2)) + t.tensor([x_base, y_base]).cuda(0)
            current_grid_target_top_left = current_grid_target_xy_true - current_grid_target_wh_true / 2
            current_grid_target_bottom_right = current_grid_target_xy_true + current_grid_target_wh_true / 2
            current_target_coord = t.cat((current_grid_target_top_left, current_grid_target_bottom_right), dim=1)
            current_grid_pred_top_left = current_grid_pred_xy_true - current_grid_pred_wh_true / 2
            current_grid_pred_bottom_right = current_grid_pred_xy_true + current_grid_pred_wh_true / 2
            current_pred_coord = t.cat((current_grid_pred_top_left, current_grid_pred_bottom_right), dim=1)

            ious = box_iou(current_target_coord, current_pred_coord)
            highest_iou_pred_index_argsort = t.argsort(ious, dim=1)
            if t.unique(highest_iou_pred_index_argsort[:, -1]).size()[0] == current_grid_obj_count:
                highest_iou_pred_index = list(highest_iou_pred_index_argsort[:, -1].view(-1).cpu().numpy())
            else:
                highest_iou_pred_index = []
                for i in range(current_grid_obj_count):
                    for j in range(self.B):
                        current_max_pred_index = highest_iou_pred_index_argsort[i, self.B - j - 1].item()
                        if current_max_pred_index not in highest_iou_pred_index:
                            highest_iou_pred_index.append(current_max_pred_index)
                            break
            highest_ious = ious[list(range(current_grid_obj_count)), highest_iou_pred_index]
            has_obj_bbox_target_conf_info = has_obj_bbox_target_conf_info * highest_ious.view(-1)

            highest_iou_pred_xy_info = current_grid_pred_xy_info.view((-1, 2))[highest_iou_pred_index, :]
            highest_iou_pred_wh_info = current_grid_pred_wh_info.view((-1, 2))[highest_iou_pred_index, :]
            highest_iou_pred_conf_info = current_grid_pred_conf_info.view((-1, 1))[highest_iou_pred_index, [0] * current_grid_obj_count].view(-1)
            current_grid_noobj_pred_index = list(set(list(range(self.B))) - set(highest_iou_pred_index))
            pred_noobj_conf_info = current_grid_pred_conf_info.view((-1, 1))[current_grid_noobj_pred_index, [0] * (self.B - current_grid_obj_count)].view(-1)
            loss_grid = self.calc_one_grid_bbox_loss(highest_iou_pred_xy_info, highest_iou_pred_wh_info, highest_iou_pred_conf_info, pred_noobj_conf_info, current_grid_pred_class_info, has_obj_bbox_target_xy_info, has_obj_bbox_target_wh_info, has_obj_bbox_target_conf_info, current_grid_target_class_info)
            loss_grid_all = loss_grid_all + loss_grid
        loss = (loss_noobj_conf + loss_grid_all) / targets.size()[0]
        return loss

    def calc_one_grid_bbox_loss(self, highest_iou_pred_xy_info, highest_iou_pred_wh_info, highest_iou_pred_conf_info, pred_noobj_conf_info, current_grid_pred_class_info, has_obj_bbox_target_xy_info, has_obj_bbox_target_wh_info, has_obj_bbox_target_conf_info, current_grid_target_class_info):
        loss_class = t.sum((current_grid_pred_class_info - current_grid_target_class_info) ** 2)
        loss_xy = t.sum((has_obj_bbox_target_xy_info.view((-1, 2)) - highest_iou_pred_xy_info) ** 2) * self.lamda_coord
        loss_wh = t.sum((has_obj_bbox_target_wh_info.view((-1, 2)) ** 0.5 - highest_iou_pred_wh_info ** 0.5) ** 2) * self.lamda_coord
        loss_conf_obj = t.sum((has_obj_bbox_target_conf_info - highest_iou_pred_conf_info) ** 2)
        loss_conf_noobj = t.sum(pred_noobj_conf_info ** 2) * self.lamda_noobj
        return loss_class + loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj

    def calc_noobj_conf_loss(self, noobj_bbox_pred_conf):
        return self.lamda_noobj * t.sum((noobj_bbox_pred_conf) ** 2)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = t.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = t.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

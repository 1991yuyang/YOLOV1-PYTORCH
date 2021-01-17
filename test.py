from model import YOLORES, ORIYOLO
import numpy as np
import torch as t
import cv2
from numpy import random as rd
from torchvision.ops import nms
import os


def read_one_img(img_pth, img_size, use_cuda):
    orig_img = cv2.imread(img_pth)
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("a", img)
    # cv2.waitKey()
    orig_img_height = img.shape[0]
    orig_img_width = img.shape[1]
    img = cv2.resize(img, (img_size, img_size))
    img = t.tensor(np.transpose(img / 255, axes=[2, 0, 1])).type(t.FloatTensor).unsqueeze(0)
    if use_cuda:
        return img.cuda(0), orig_img, (orig_img_height, orig_img_width)
    return img, orig_img, (orig_img_height, orig_img_width)


def load_model(model_save_path, S, B, num_classes, use_cuda, backbone_use_resnet):
    if backbone_use_resnet:
        YOLO = YOLORES
    else:
        YOLO = ORIYOLO
    model = YOLO(S, B, num_classes)
    if use_cuda:
        model = model.cuda(0)
    model.load_state_dict(t.load(model_save_path))
    model.eval()
    return model


def predict_one_img(model, img, orig_img, S, B, classes, confidence_thresh, orig_image_size, use_cuda, iou_thresh, colors):
    grid_height = orig_image_size[0] / S
    grid_width = orig_image_size[1] / S
    with t.no_grad():
        model_output = model(img)[0]
    xy_info = model_output[:, :, :B * 2]  # shape like (S, S, B * 2)
    wh_info = model_output[:, :, B * 2:B * 4]  # shape like (S, S, B * 2)
    conf_info = model_output[:, :, B * 4:B * 5]  # shape like (S, S, B)
    class_info = model_output[:, :, B * 5:]  # shape like (S, S, num_classes)
    grid_y_index, grid_x_index, channel_index = t.where(conf_info > confidence_thresh)


    grid_obj_index = np.array(tuple(set(tuple(zip(grid_y_index.cpu().numpy(), grid_x_index.cpu().numpy())))))
    if len(grid_obj_index) == 0:
        print("no object detected in image")
        return None, orig_img
    grid_y_index_obj = grid_obj_index[:, 0]
    grid_x_index_obj = grid_obj_index[:, 1]
    grid_obj_base_x_y = np.array([(i[1] * grid_width, i[0] * grid_height) for i in grid_obj_index])
    grid_obj_base_x_y = t.tensor(np.concatenate([grid_obj_base_x_y] * B, axis=1)).type(t.FloatTensor)
    if use_cuda:
        grid_obj_base_x_y = grid_obj_base_x_y.cuda(0)
    grid_obj_xy = xy_info[grid_y_index_obj, grid_x_index_obj, :]
    if use_cuda:
        grid_obj_add_x_y = grid_obj_xy * t.tensor([grid_width, grid_height] * B).cuda(0)
    else:
        grid_obj_add_x_y = grid_obj_xy * t.tensor([grid_width, grid_height] * B)
    grid_obj_true_xy = grid_obj_base_x_y + grid_obj_add_x_y
    grid_obj_wh = wh_info[grid_y_index_obj, grid_x_index_obj, :]
    if use_cuda:
        grid_obj_true_wh = grid_obj_wh * t.tensor([orig_image_size[1], orig_image_size[0]] * B).type(t.FloatTensor).cuda(0)
    else:
        grid_obj_true_wh = grid_obj_wh * t.tensor([orig_image_size[1], orig_image_size[0]] * B).type(t.FloatTensor)
    grid_obj_conf = conf_info[grid_y_index_obj, grid_x_index_obj, :]
    class_obj = class_info[grid_y_index_obj, grid_x_index_obj, :]
    grid_obj_class_index = t.argmax(class_obj, dim=1)
    grid_obj_class_index_unique = np.unique(grid_obj_class_index.cpu().numpy())
    grid_obj_class_prob = t.max(class_obj, dim=1).values
    grid_obj_top_left = grid_obj_true_xy - grid_obj_true_wh / 2
    grid_obj_bottom_right = grid_obj_true_xy + grid_obj_true_wh / 2
    top_left_of_every_B = []
    bottom_right_of_every_B = []
    conf_of_every_B = []
    class_index_of_every_B = []
    class_prob_of_every_B = []
    for i in range(B):
        class_index_of_every_B.append(grid_obj_class_index.view((-1, 1)))
        class_prob_of_every_B.append(grid_obj_class_prob.view(-1, 1))
        top_left_of_every_B.append(grid_obj_top_left[:, i * 2:(i + 1) * 2])
        bottom_right_of_every_B.append(grid_obj_bottom_right[:, i * 2:(i + 1) * 2])
        conf_of_every_B.append(grid_obj_conf[:, i].view((-1, 1)))
    class_prob_of_every_B = t.cat(tuple(class_prob_of_every_B), dim=0)
    top_left_of_every_B = t.cat(tuple(top_left_of_every_B), dim=0)
    bottom_right_of_every_B = t.cat(tuple(bottom_right_of_every_B), dim=0)

    class_index_of_every_B = t.cat(tuple(class_index_of_every_B), dim=0)
    conf_of_every_B = t.cat(tuple(conf_of_every_B), dim=0)
    class_prob_of_every_B = conf_of_every_B * class_prob_of_every_B
    coord_of_every_B = t.cat((top_left_of_every_B, bottom_right_of_every_B), dim=1)
    result = {}
    for class_index in grid_obj_class_index_unique:
        color = colors[class_index]
        curr_class = {}
        curr_class_name = classes[class_index]
        curr_class_B_index = class_index_of_every_B == class_index
        # conf_of_B_of_curr_class = conf_of_every_B[curr_class_B_index.view(-1), :]
        class_prob_of_B_of_curr_class = class_prob_of_every_B[curr_class_B_index.view(-1), :].view(-1)
        coord_of_B_of_curr_class = coord_of_every_B[curr_class_B_index.view(-1), :]
        curr_class_keeped_B_index = nms(coord_of_B_of_curr_class, class_prob_of_B_of_curr_class, iou_thresh)
        coord_after_nms = coord_of_B_of_curr_class[curr_class_keeped_B_index, :]
        class_prob_after_nms = class_prob_of_B_of_curr_class[curr_class_keeped_B_index]
        curr_class["coord"] = coord_after_nms
        curr_class["class_prob"] = class_prob_after_nms
        result[curr_class_name] = curr_class
        # draw bounding box on orig img
        for i in range(class_prob_after_nms.size()[0]):
            coord = coord_after_nms[i]
            class_prob = class_prob_after_nms[i].item()
            cv2.rectangle(orig_img, tuple(coord[:2].detach().cpu().numpy()), tuple(coord[2:].detach().cpu().numpy()), (int(color[0]), int(color[1]), int(color[2])), 2)
            cv2.putText(orig_img, curr_class_name + ':' + "%.2f" % (class_prob,), tuple(coord[:2].detach().cpu().numpy()), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (int(color[0]), int(color[1]), int(color[2])), 2)
    # cv2.imshow("result", orig_img)
    # cv2.waitKey()
    return result, orig_img


def main(common_conf, test_conf):
    S = common_conf["S"]
    B = common_conf["B"]
    num_classes = common_conf["num_classes"]
    classes = common_conf["classes"]
    best_model_save_path = common_conf["best_model_save_path"]
    epoch_model_save_path = common_conf["epoch_model_save_path"]
    backbone_use_resnet = bool(common_conf["backbone_use_resnet"])
    img_size = common_conf["img_size"]
    test_set_dir = test_conf["test_set_dir"]
    output_result_dir = test_conf["output_result_dir"]
    nms_iou_thresh = test_conf["nms_iou_thresh"]
    confidence_thresh = test_conf["confidence_thresh"]
    use_cuda = bool(test_conf["use_cuda"])
    use_best_model = bool(test_conf["use_best_model"])
    img_paths = [[os.path.join(test_set_dir, i), i] for i in os.listdir(test_set_dir)]

    if use_best_model:
        print("load %s" % (best_model_save_path,))
        model = load_model(best_model_save_path, S, B, num_classes, use_cuda, backbone_use_resnet)
    else:
        print("load %s" % (epoch_model_save_path,))
        model = load_model(epoch_model_save_path, S, B, num_classes, use_cuda, backbone_use_resnet)
    color_r = tuple(rd.permutation(np.linspace(60, 255, num_classes).astype(np.int)))
    color_g = tuple(rd.permutation(np.linspace(80, 255, num_classes).astype(np.int)))
    color_b = tuple(rd.permutation(np.linspace(100, 255, num_classes).astype(np.int)))
    colors = list(zip(color_r, color_g, color_b))
    for img_path, i in img_paths:
        print("predict %s" % (i,))
        img, orig_img, orig_image_size = read_one_img(img_path, img_size, use_cuda)
        result, img_with_bbox = predict_one_img(model, img, orig_img, S, B, classes, confidence_thresh, orig_image_size, use_cuda, nms_iou_thresh, colors)
        cv2.imwrite(os.path.join(output_result_dir, i), img_with_bbox)


if __name__ == "__main__":
    with open("./conf.json", "r", encoding="utf-8") as file:
        conf = eval(file.read())
    common_conf = conf["common"]
    test_conf = conf["test"]
    main(common_conf, test_conf)
    # S = common_conf["S"]
    # B = common_conf["B"]
    # num_classes = common_conf["num_classes"]
    # classes = common_conf["classes"]
    # best_model_save_path = common_conf["best_model_save_path"]
    # img_size = common_conf["img_size"]
    # test_set_dir = test_conf["test_set_dir"]
    # output_result_dir = test_conf["output_result_dir"]
    # nms_iou_thresh = test_conf["nms_iou_thresh"]
    # confidence_thresh = test_conf["confidence_thresh"]
    # use_cuda = bool(test_conf["use_cuda"])
    #
    #
    # img_path = "/home/yuyang/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"
    # classes = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
    # img, orig_img, orig_image_size = read_one_img(img_path, 448, True)
    # model = load_model("./best.pth", 7, 2, 20, True)
    # result = predict_one_img(model, img, orig_img, 7, 2, classes, 0.4, orig_image_size, True, 0.2)
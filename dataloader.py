from torch.utils import data
import os
import cv2
from lxml import etree
import numpy as np
import torch as t
from loss import YOLOLoss
from model import YOLORES


class YoloSet(data.Dataset):

    def __init__(self, B, S, img_type, mode, voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size):
        """

        :param B: every grid cell predict B bounding boxes
        :param S: every image is splited become S * S grid cell
        :param img_type: jpg or png
        :param mode: train or val
        :param voc_Annotations_dir: xml file dir
        :param voc_Main_dir: txt file dir, include train.txt and val.txt
        :param voc_JPEGImages_dir: image file dir
        :param classes: class names, list type for example ["person", "horse", ....]
        :param img_size: image size
        """
        assert img_type.lower() in ["jpg", "png"], "img_type should be jpg or png"
        assert mode.lower() in ["train", "val"], "mode should be train or val"
        self.voc_Annotations_dir = voc_Annotations_dir
        self.voc_Main_dir = voc_Main_dir
        assert os.path.exists(os.path.join(voc_Main_dir, "train.txt")) and os.path.exists(os.path.join(voc_Main_dir, "val.txt")), "train.txt and val.txt should under voc_Main_dir"
        self.voc_JPEGImages_dir = voc_JPEGImages_dir
        if mode.lower() == "train":
            with open(os.path.join(self.voc_Main_dir, "train.txt"), "r", encoding="utf-8") as file:
                img_names = file.read().strip("\n").split("\n")
        if mode.lower() == "val":
            with open(os.path.join(self.voc_Main_dir, "val.txt"), "r", encoding="utf-8") as file:
                img_names = file.read().strip("\n").split("\n")
        self.img_paths = [os.path.join(self.voc_JPEGImages_dir, img_name + "." + img_type.lower()) for img_name in img_names]
        self.xml_paths = [os.path.join(self.voc_Annotations_dir, img_name + ".xml") for img_name in img_names]
        self.S = S
        self.B = B
        self.classes = classes
        self.img_size = img_size

    def __getitem__(self, index):
        label = np.zeros((self.S, self.S, self.B * 5 + len(self.classes)), dtype=np.float64)  # shape like (S, S, B * 5 + num_classes), every grid cell represent: [x1, y1, x2, y2, ..., xB, yB, w1, h1, w2, h2, ..., wB, hB, c1, c2, ..., cB, p_1, p_2, ..., p_num_classes]
        img_path = self.img_paths[index]
        xml_path = self.xml_paths[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_height = img.shape[0]
        img_width = img.shape[1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img / 255, axes=[2, 0, 1])
        grid_cell_width = img_width / self.S
        grid_cell_height = img_height / self.S
        tree = etree.parse(xml_path)
        root = tree.getroot()
        object_nodes = root.findall("object")
        for object_node in object_nodes:
            # iter every objects in current image
            object_name = object_node.find("name").text  # current object's class
            class_index = self.classes.index(object_name)  # current object' class index in classes
            bndbox_node = object_node.find("bndbox")
            xmin = float(bndbox_node.find("xmin").text)
            ymin = float(bndbox_node.find("ymin").text)
            xmax = float(bndbox_node.find("xmax").text)
            ymax = float(bndbox_node.find("ymax").text)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            x_grid_index = int(center_x // grid_cell_width)
            y_grid_index = int(center_y // grid_cell_height)
            x_to_label = center_x / grid_cell_width - center_x // grid_cell_width
            y_to_label = center_y / grid_cell_height - center_y // grid_cell_height
            w_to_label = (xmax - xmin) / img_width
            h_to_label = (ymax - ymin) / img_height
            curr_grid_obj_count = int(np.sum(label[y_grid_index, x_grid_index, self.B * 4:self.B * 5]))
            curr_grid_class_info = label[y_grid_index, x_grid_index, self.B * 5:]
            if curr_grid_obj_count < self.B:
                if curr_grid_obj_count > 1:
                    if curr_grid_class_info[class_index] != 1:
                        continue
                else:
                    label[y_grid_index, x_grid_index, self.B * 5 + class_index] = 1
                label[y_grid_index, x_grid_index, curr_grid_obj_count * 2] = x_to_label
                label[y_grid_index, x_grid_index, curr_grid_obj_count * 2 + 1] = y_to_label
                label[y_grid_index, x_grid_index, self.B * 2 + curr_grid_obj_count * 2] = w_to_label
                label[y_grid_index, x_grid_index, self.B * 2 + curr_grid_obj_count * 2 + 1] = h_to_label
                label[y_grid_index, x_grid_index, self.B * 4 + curr_grid_obj_count] = 1
        return t.tensor(img).type(t.FloatTensor), t.tensor(label).type(t.FloatTensor), t.tensor([img_height, img_width]).type(t.FloatTensor)

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    B = 2
    S = 7
    img_type = "jpg"
    mode = "train"
    voc_Annotations_dir = "/home/yuyang/dataset/VOCdevkit/VOC2012/Annotations"
    voc_Main_dir = "/home/yuyang/dataset/VOCdevkit/VOC2012/ImageSets/Main"
    voc_JPEGImages_dir = "/home/yuyang/dataset/VOCdevkit/VOC2012/JPEGImages"
    classes = ["bottle", "cow", "sofa", "pottedplant", "boat", "car", "bus", "chair", "cat", "aeroplane", "person", "sheep", "diningtable", "train", "tvmonitor", "motorbike", "bird", "dog", "horse", "bicycle"]
    img_size = 448
    s = data.DataLoader(YoloSet(B, S, img_type, mode, voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size), batch_size=4, shuffle=True, drop_last=False)
    criterion = YOLOLoss(S, B, len(classes), 5, 0.5)
    model = YOLORES(S, B, len(classes)).cuda(0)
    for img, label, orig_img_size in s:
        model_output = model(img.cuda(0))
        criterion(model_output.cuda(0), label.cuda(0), orig_img_size.cuda(0))




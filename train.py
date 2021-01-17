from model import ORIYOLO, YOLORES
from loss import YOLOLoss
import torch as t
from torch import nn, optim
import os
from dataloader import YoloSet
from torch.utils import data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_step(model, d_train, l_train, orig_image_sizes, optimizer, criterion):
    model.train()
    d_train_cuda = d_train.cuda(0)
    l_train_cuda = l_train.cuda(0)
    orig_image_sizes_cuda = orig_image_sizes.cuda(0)
    train_output = model(d_train_cuda)
    train_loss = criterion(train_output, l_train_cuda, orig_image_sizes_cuda)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    return model, train_loss.item()


def valid_step(model, valid_loader, criterion, batch_size, B, S, image_type, voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size, num_workers):
    model.eval()
    try:
        d_valid, l_valid, orig_image_sizes = next(valid_loader)
    except:
        valid_loader = iter(data.DataLoader(YoloSet(B, S, image_type, "val", voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers))
        d_valid, l_valid, orig_image_sizes = next(valid_loader)
    d_valid_cuda = d_valid.cuda(0)
    l_valid_cuda = l_valid.cuda(0)
    orig_image_sizes_cuda = orig_image_sizes.cuda(0)
    with t.no_grad():
        valid_output = model(d_valid_cuda)
    valid_loss = criterion(valid_output, l_valid_cuda, orig_image_sizes_cuda)
    return model, valid_loss.item()


def train(train_conf, common_conf):
    current_best_valid_loss = float("inf")
    voc_Annotations_dir = train_conf["voc_Annotations_dir"]
    voc_Main_dir = train_conf["voc_Main_dir"]
    voc_JPEGImages_dir = train_conf["voc_JPEGImages_dir"]
    lamda_coord = train_conf["lamda_coord"]
    lamda_noobj = train_conf["lamda_noobj"]
    img_size = common_conf["img_size"]
    lr = train_conf["lr"]
    batch_size = train_conf["batch_size"]
    epoch = train_conf["epoch"]
    weight_decay = train_conf["weight_decay"]
    num_workers = train_conf["dataloader_num_workers"]
    lr_shrink_rate = train_conf["lr_shrink_rate"]
    lr_shrink_epoch = train_conf["lr_shrink_epoch"]
    best_model_save_path = common_conf["best_model_save_path"]
    epoch_model_save_path = common_conf["epoch_model_save_path"]
    image_type = train_conf["image_type"]
    B = common_conf["B"]
    S = common_conf["S"]
    num_classes = common_conf["num_classes"]
    classes = common_conf["classes"]
    backbone_use_resnet = bool(common_conf["backbone_use_resnet"])
    if backbone_use_resnet:
        YOLO = YOLORES
    else:
        YOLO = ORIYOLO
    model = YOLO(S, B, num_classes)
    ########################################
    start_e = 1
    if os.path.exists("./continue_train.txt"):
        print("continue training load epoch model......")
        model.load_state_dict(t.load(epoch_model_save_path))
        with open("./continue_train.txt", "r", encoding="utf-8") as file:
            start_e, current_best_valid_loss, lr = file.read().split(" ")
        start_e = int(start_e)
        current_best_valid_loss = float(current_best_valid_loss)
        lr = float(lr)
    #########################################
    model = nn.DataParallel(module=model, device_ids=[0])
    model = model.cuda(0)
    criterion = YOLOLoss(S, B, num_classes, lamda_coord, lamda_noobj)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
    for e in range(start_e, 1 + epoch):
        train_loader = iter(data.DataLoader(YoloSet(B, S, image_type, "train", voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers))
        valid_loader = iter(data.DataLoader(YoloSet(B, S, image_type, "val", voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers))
        all_steps = len(train_loader)
        current_step = 0
        for d_train, l_train, orig_image_sizes in train_loader:
            model, train_loss = train_step(model, d_train, l_train, orig_image_sizes, optimizer, criterion)
            model, valid_loss = valid_step(model, valid_loader, criterion, batch_size, B, S, image_type, voc_Annotations_dir, voc_Main_dir, voc_JPEGImages_dir, classes, img_size, num_workers)
            current_step += 1
            print("epoch:%d/%d, step:%d/%d, train_loss:%.5f, valid_loss:%.5f" % (e, epoch, current_step, all_steps, train_loss, valid_loss))
            if valid_loss < current_best_valid_loss:
                current_best_valid_loss = valid_loss
                print("saving best model......")
                t.save(model.module.state_dict(), best_model_save_path)

        print("saving epoch model......")
        t.save(model.module.state_dict(), epoch_model_save_path)
        if e % lr_shrink_epoch == 0:
            lr *= lr_shrink_rate
            optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        ##############################################################
        with open("./continue_train.txt", "w", encoding="utf-8") as file:
            file.write("%d %f %.10f" % (e + 1, current_best_valid_loss, lr))
        ##############################################################


if __name__ == "__main__":
    with open("./conf.json", "r", encoding="utf-8") as file:
        conf = eval(file.read())
    train_conf = conf["train"]
    common_conf = conf["common"]
    train(train_conf, common_conf)
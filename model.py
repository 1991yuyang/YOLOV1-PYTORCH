import torch as t
from torch import nn
from torchvision import models


class Conv3X3(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(Conv3X3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)


class Conv1X1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv1X1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)


class ORIYOLO(nn.Module):

    def __init__(self, S, B, num_classes):
        """

        :param S: split every image into S * S grid cell
        :param B: bounding box count that every grid cell predict
        :param num_classes: count of object classes
        """
        super(ORIYOLO, self).__init__()
        self.B = B
        self.S = S
        self.num_classes = num_classes
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block_2 = nn.Sequential(
            Conv3X3(in_channels=64, out_channels=192, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block_3 = nn.Sequential(
            Conv1X1(in_channels=192, out_channels=128),
            Conv3X3(in_channels=128, out_channels=256, stride=1),
            Conv1X1(in_channels=256, out_channels=256),
            Conv3X3(in_channels=256, out_channels=512, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block_4 = nn.Sequential(
            Conv1X1(in_channels=512, out_channels=256),
            Conv3X3(in_channels=256, out_channels=512, stride=1),
            Conv1X1(in_channels=512, out_channels=256),
            Conv3X3(in_channels=256, out_channels=512, stride=1),
            Conv1X1(in_channels=512, out_channels=256),
            Conv3X3(in_channels=256, out_channels=512, stride=1),
            Conv1X1(in_channels=512, out_channels=256),
            Conv3X3(in_channels=256, out_channels=512, stride=1),
            Conv1X1(in_channels=512, out_channels=512),
            Conv3X3(in_channels=512, out_channels=1024, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block_5 = nn.Sequential(
            Conv1X1(in_channels=1024, out_channels=512),
            Conv3X3(in_channels=512, out_channels=1024, stride=1),
            Conv1X1(in_channels=1024, out_channels=512),
            Conv3X3(in_channels=512, out_channels=1024, stride=1),
            Conv3X3(in_channels=1024, out_channels=1024, stride=1),
            Conv3X3(in_channels=1024, out_channels=1024, stride=2)
        )
        self.block_6 = nn.Sequential(
            Conv3X3(in_channels=1024, out_channels=1024, stride=1),
            Conv3X3(in_channels=1024, out_channels=1024, stride=1)
        )
        self.last = nn.Sequential(
            nn.Upsample(size=(self.S, self.S), mode="bilinear"),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=self.B * 5 + self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.B * 5 + self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.last(x)
        x = x.permute(dims=(0, 2, 3, 1))  # output shape like (N, S, S, B * 5 + num_classes), every grid cell represent: [x1, y1, x2, y2, ..., xB, yB, w1, h1, w2, h2, ..., wB, hB, c1, c2, ..., cB, p_1, p_2, ..., p_num_classes]
        return x


class YOLORES(nn.Module):

    def __init__(self, S, B, num_classes):
        """

                :param S: split every image into S * S grid cell
                :param B: bounding box count that every grid cell predict
                :param num_classes: count of object classes
                """
        super(YOLORES, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.last = nn.Sequential(
            nn.Upsample(size=(self.S, self.S), mode="bilinear"),
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=self.B * 5 + self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.B * 5 + self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.last(x)
        x = x.permute(dims=(0, 2, 3, 1))  # output shape like (N, S, S, B * 5 + num_classes), every grid cell represent: [x1, y1, x2, y2, ..., xB, yB, w1, h1, w2, h2, ..., wB, hB, c1, c2, ..., cB, p_1, p_2, ..., p_num_classes]
        return x


if __name__ == "__main__":
    model = YOLORES(7, 2, 20)
    d = t.randn(1, 3, 448, 448)
    model.eval()
    with t.no_grad():
        output = model(d)
    print(output.size())
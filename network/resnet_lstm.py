import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 定义resnet50

class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet50Custom, self).__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=num_classes), nn.Tanh())

    def forward(self, x):
        x = self.resnet(x)
        return x


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # print('shape of out', out.size())  # batch size * 4 * 48 (hidden size)
        out = self.fc(out[:, -1, :])
        # print('shape of out', out.size())  # batch size * 12
        return out


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.resnet = ResNet50Custom(num_classes=12)
        self.lstm = CustomLSTM(input_size=12, hidden_size=48, num_layers=2, num_classes=12)
        self.fc01 = nn.Linear(2, 12)
        self.fc02 = nn.Linear(24, 12)

    def forward(self, x_img, x_coordinate):
        results = []
        for i in range(4):
            img_i = self.resnet(x_img[:, i, :, :, :])
            # print('shape of img_i', img_i.size()) # 4*12
        # 处理每张图片的坐标信息，并把它和图片的处理结果相结合
            # coordinate_i = self.fc01(x_coordinate[:, (2 * i): (2 * i + 2)])
            # print('shape of coor in', x_coordinate.size())  # 4*4*12
            coordinate_i = self.fc01(x_coordinate[:, i, :])
            # print('shape of coor i', coordinate_i.size())  # 4*12
            results_i = torch.cat((img_i, coordinate_i), dim=1)
            # print('shape of result i', results_i.size())  # 4*24
            results_i = self.fc02(results_i)  # size (batch_size, 12)
            results.append(results_i)
        

        # 将四个结果合并成一个batch
        results = torch.stack(results, dim=1)  # size: (batch_size, 4, 12)
        # print('shape of results', results.size())  # 4*4*12

        # 输入到LSTM网络
        lstm_out = self.lstm(results)

        # print('shape of lstm out', lstm_out.size())  # batchsize * 12

        output = lstm_out.reshape(-1, 6, 2)

        # print('shape of output', output.size())  # batch size * 6 * 2

        return output

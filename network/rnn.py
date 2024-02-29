import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 定义resnet50

class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=1000):
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
        self.resnet = ResNet50Custom(num_classes=1000)
        self.lstm = CustomLSTM(input_size=12, hidden_size=48, num_layers=2, num_classes=12)
        self.fc01 = nn.Linear(2, 1000)
        self.fc02 = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500,100),
            nn.Linear(100, 12),
            nn.BatchNorm1d(12)
        )

    def forward(self, x_img, x_coordinate):
        size_img = x_img.size()  # batch * 4 * image
        size_coo = x_coordinate.size()  # batch * 4 * 2

        #  merge the first and second dimension to avoid recurring function
        img_new = x_img.reshape(size_img[0] * size_img[1], *size_img[2:])  # 4*batch * image
        coo_new = x_coordinate.reshape(size_coo[0] * size_coo[1], *size_coo[2:])  # 4*batch * 2

        # process images and coordinates
        x1 = self.resnet(img_new)  # 4*batch * 1000
        # print('shape of x1')
        # print(x1.size())
        x2 = self.fc01(coo_new)  # 4*batch * 1000
        # print('shape of x2')
        # print(x2.size())
        x3 = torch.cat((x1, x2), dim=1)  # concatinate x1 and x2 to get 4*batch * 2000
        x3 = self.fc02(x3)  # 4*batch * 12
        num_node = x3.size(1)
        x4 = x3.view(x_coordinate.size(0), x_coordinate.size(1), num_node)  # batch * 4 * 12

        # 输入到LSTM网络
        lstm_out = self.lstm(x4)

        # print('shape of lstm out', lstm_out.size())  # batchsize * 12

        output = lstm_out.reshape(-1, 6, 2)

        # print('shape of output', output.size())  # batch size * 6 * 2

        return output

import sys
import os
import argparse
import torch
from resnet_onecamera import MyNetwork
from torch.utils.data import DataLoader
from data_preparation.resnet_data import SimulatorDataset
import utils
import cv2
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_val(cfg):
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 第1步：构建数据读取迭代器

    train_data = SimulatorDataset(data_path=cfg.train_data)
    val_data = SimulatorDataset(data_path=cfg.val_data)

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = MyNetwork().to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络

    val_losses = []
    min_loss = 100
    min_epoch = 0

    for epoch_i in tqdm(range(cfg.epochs), 'Training processing'):
        # print('number of epochs', epoch_i)
        model.train()
        train_loss = 0.0

        for train_i, (input_img, input_xy, target) in enumerate(train_dataloader):

            batchsize = input_img.shape[0]
            input_new = torch.rand(batchsize, 3, 224, 224)

            for i in range(batchsize):
                for group_img in range(4):
                    img = input_img[i, group_img].permute(1, 2, 0)
                    img = img.numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = utils.preprocess(img)
                    img = utils.random_shadow(img)
                    img = utils.random_brightness(img)
                    input_new[i, group_img] = torch.from_numpy(img).permute(2, 0, 1)

            input_new, input_xy, target = input_new.to(device), input_xy.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input_new, input_xy)
            loss = loss_function(output, target)
            train_loss = train_loss + loss

            # print('loss of train in batch %d' % train_i, loss.item())

            loss.backward()
            optimizer.step()

        print('train loss:', train_loss.item() / (train_i + 1))

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            for val_i, (input_img, input_xy, target) in enumerate(val_dataloader):
                input_img, input_xy, target = input_new.to(device), input_xy.to(device), target.to(device)
                output = model(input_new)
                loss_sum = loss_sum + loss_function(output, target)

            val_loss = loss_sum.item() / (val_i + 1)
            if val_loss < min_loss:
                min_loss = val_loss
                min_epoch = epoch_i
            print(val_i)
            print('val_loss:', val_loss)

        val_losses.append(val_loss)

        torch.save(model, cfg.save_path + '/' + str(epoch_i) + '.pt')
        torch.cuda.empty_cache()

    plt.plot(np.arange(len(val_losses)), val_losses, label='validation loss')
    plt.savefig('val_loss(%d,%f,%d).png' % (cfg.epochs, min_loss, min_epoch))
    # plt.show()
    # plt.clf()

def test(cfg):
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('gpu:0')

        # 第1步：构建数据读取迭代器

    test_data = SimulatorDataset(data_path=cfg.test_data)

    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = MyNetwork()
    model = torch.load('weights/resnet/349.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络

    for epoch_i in range(cfg.epochs):
        model.train()

        for train_i, (input, target) in enumerate(train_dataloader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = loss_function(output, target)

            print(loss.item())

            loss.backward()
            optimizer.step()

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            for val_i, (input, target) in enumerate(val_dataloader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss_sum = loss_sum + loss_function(output, target)
            print(val_i)
            print('val_loss:', loss_sum.item() / (val_i + 1))

        torch.save(model, cfg.save_path + '/' + str(epoch_i) + '.pt')
        torch.cuda.empty_cache()


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='total number of training epochs')
    parser.add_argument('--train-data', type=str, default='data/simulator/Dataset1/VehicleData.txt', help='data path')
    parser.add_argument('--val-data', type=str, default='data/simulator/Dataset/VehicleData.txt', help='data path')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet', help='path to save checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()

    train_val(cfg)


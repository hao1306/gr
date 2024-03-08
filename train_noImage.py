import sys
import os
import argparse
import torch
from network.rnn import MyNetwork, MyNetwork_LSTM, MyNetwork_NOimage, MyNetwork_LSTM_NOimage
from torch.utils.data import DataLoader, random_split
from dataset.dataloader01 import ResNetDataset, ResNetDataset_NOimage
import utils
import cv2
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
from torch.optim.lr_scheduler import StepLR


def train_val(cfg, load_model, device0):
    device = device0

    # 第1步：构建数据读取迭代器

    train_set = ResNetDataset_NOimage(cfg, phase='train')
    val_set = ResNetDataset_NOimage(cfg, phase='val')

    # 创建数据加载器
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)
    
    # print('len of train', len(train_dataloader))
    # print('len of val', len(val_dataloader))

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.L1Loss()

    scheduler = StepLR(optimizer, step_size=3, gamma=0.9)

    # 第3步：循环读取数据训练网络

    val_losses = ['validation loss']
    train_losses = ['train loss']
    save_info = [
        ['min epoch of train loss', 0],
        ['min train loss', 100],
        ['learning rate', 0],
        ['min epoch of val loss', 0],
        ['min val loss', 100],
        ['learning rate', 0]
    ]

    for epoch_i in range(cfg.epochs):
        # print('number of epochs', epoch_i)
        model.train()
        sum_train = 0.0

        for (input_xy, target, coo_original) in tqdm(train_dataloader, desc=f'train Epoch {epoch_i} / {cfg.epochs}' ):
            time.sleep(0.00001)
            input_xy, target = input_xy.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(input_xy)
            loss = loss_function(output, target)
            sum_train += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = sum_train / (len(train_dataloader))
        scheduler.step()

        print(f'The train loss of epoch {epoch_i} is :', train_loss)
        # print('len(train_dataloader)', len(train_dataloader))
        train_losses.append(train_loss)

        if train_loss < save_info[1][1]:
            save_info[1][1] = train_loss
            save_info[0][1] = epoch_i
            save_info[2][1] = optimizer.param_groups[0]['lr']

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            for (input_xy, target, coo_original) in tqdm(val_dataloader, desc=f'val Epoch {epoch_i} / {cfg.epochs}'):
                input_xy, target = input_xy.to(device), target.to(device)
                output = model(input_xy)
                loss_sum += loss_function(output, target).item()

            val_i = len(val_dataloader)
            val_loss = loss_sum / val_i
            if val_loss < save_info[4][1]:
                save_info[4][1] = val_loss
                save_info[3][1] = epoch_i
                save_info[5][1] = optimizer.param_groups[0]['lr']

        val_losses.append(val_loss)

        print(f'The val loss of epoch {epoch_i} is :', val_loss)

        save_path = cfg.save_path + '/' + str(epoch_i) + '.pt'
        torch.save(model, save_path)
        torch.cuda.empty_cache()
    
    csv_file = cfg.save_path + '/' + 'result report of train'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(save_info)
    
    csv_file1 = cfg.save_path + '/' + 'loss'
    all_loss = [train_losses, val_losses]
    all_loss = list(map(list, zip(*all_loss)))
    with open(csv_file1, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_loss)

    plt.figure()
    plt.plot(np.arange(len(val_losses) - 1), np.array(train_losses[1:]).astype(float), label='train loss')
    plt.plot(np.arange(len(val_losses) - 1), np.array(val_losses[1:]).astype(float), label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.savefig(cfg.save_path + '/' + 'train_val_loss.png')



def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='total number of training epochs')
    parser.add_argument('--epoch_test', type=int, default=20, help='total number of testing epochs')
    
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/one_camera/try07_LSTM_noImage_L1', help='path to save checkpoint')

    parser.add_argument('--data-root', type=str, default='dataset/nuscenes/trian/', help='path to save checkpoint')
    parser.add_argument('--data-path', type=str, default='dataset/nuscenes/image_scene/', help='path to save checkpoint')
    parser.add_argument('--image-n', type=int, default=1, help='num of cameras to be used')
    parser.add_argument('--input-n', type=int, default=4, help='num of images as input')
    parser.add_argument('--predict-n', type=int, default=6, help='num of points to be predicted')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='path to save checkpoint')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='path to save checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    device = torch.device('cuda:0')
    if os.path.exists(cfg.save_path):
        pass
    else:
        os.mkdir(cfg.save_path)
    # load_model = torch.load('weights/one_camera/try04/19.pt')
    load_model = MyNetwork_LSTM_NOimage()
    # total_params = sum(p.numel() for p in load_model.parameters())
    # print(f"Total parameters: {total_params}")
    # torch.cuda.empty_cache()

    train_val(cfg, load_model, device)
    # test(cfg, load_model, device)


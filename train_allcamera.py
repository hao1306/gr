import sys
import os
import argparse
import torch
from network.resnet_onecamera import MyNetwork
from torch.utils.data import DataLoader, random_split
from dataset.dataloader_old import Dataset_allcamera, Dataset_allcamera_test
import utils
import cv2
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
from torch.optim.lr_scheduler import StepLR


def train_val(cfg, load_model, device1):
    device = device1

    # 第1步：构建数据读取迭代器

    dataset = Dataset_allcamera(data_path=cfg.train_data)
    # dataset, _ = random_split(dataset, [100, len(dataset) - 100])

    # 定义训练集和验证集的比例
    train_ratio = 0.8  # 80% 用于训练
    val_ratio = 0.2    # 20% 用于验证

    # 计算分割点
    num_data = len(dataset)
    num_train = int(train_ratio * num_data)
    num_val = num_data - num_train

    # 使用random_split分割数据集
    train_set, val_set = random_split(dataset, [num_train, num_val])

    # 创建数据加载器
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()

    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

    # 第3步：循环读取数据训练网络

    val_losses = ['validation loss']
    train_losses = ['train loss']
    save_info = [
        ['min epoch of train loss', 0.0],
        ['min train loss', 100.0],
        ['learning rate', 0.0],
        ['min epoch of val loss', 0.0],
        ['min val loss', 100.0],
        ['learning rate', 0.0]
    ]

    for epoch_i in range(cfg.epochs):
        # print('number of epochs', epoch_i)
        model.train()
        sum_train = 0.0

        for (input_img, input_xy, target) in tqdm(train_dataloader, desc=f'train Epoch {epoch_i} / {cfg.epochs}' ):

            time.sleep(0.0001)
            input_img, input_xy, target = input_img.to(device=device, dtype=torch.float), input_xy.to(device), target.to(device)

            optimizer.zero_grad()

            # output = model(input_new, input_xy)
            output = model(input_img, input_xy)
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
            for (input_img, input_xy, target) in tqdm(val_dataloader, desc=f'val Epoch {epoch_i} / {cfg.epochs}'):
                input_img, input_xy, target = input_img.to(device=device, dtype=torch.float), input_xy.to(device), target.to(device)
                output = model(input_img, input_xy)
                loss_sum += loss_function(output, target).item()

            val_i = len(val_dataloader)
            val_loss = loss_sum / val_i
            if val_loss < save_info[4][1]:
                save_info[4][1] = val_loss
                save_info[3][1] = epoch_i
                save_info[5][1] = optimizer.param_groups[0]['lr']

        val_losses.append(val_loss)

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
    plt.plot(np.arange(len(val_losses)), train_losses, label='train loss')
    plt.plot(np.arange(len(val_losses)), val_losses, label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.savefig(cfg.save_path + '/' + 'train_val_loss for all cameras.png')
    print('the training process is finished')



def test(cfg, model_, device):

    dataset = Dataset_allcamera_test(data_path=cfg.test_data)
    # num_data = len(dataset)
    # aa = num_data - 5
    # test_dataset, _ = random_split(dataset, [5, aa])
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    #                               pin_memory=True)

    test_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    model = model_.to(device=device)

    loss_function = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        for (input_img, input_xy, target) in tqdm(test_dataloader, "Testing"):
            time.sleep(0.001)
            input_img, input_xy, target = input_img.to(device), input_xy.to(device), target.to(device)
            output = model(input_img, input_xy)
            loss_sum = loss_sum + loss_function(output, target).item()
            # print(loss_sum)

        test_loss = loss_sum / len(test_dataloader)
    print('The test loss of model is: ', test_loss)
    torch.cuda.empty_cache()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help='total number of training epochs')
    parser.add_argument('--epoch_test', type=int, default=20, help='total number of testing epochs')
    parser.add_argument('--train-data', type=str, default='dataset/nuscenes/trian/data_all_camera.csv', help='data path')
    # parser.add_argument('--train-data', type=str, default='dataset/nuscenes/test/data_all_camera_test.csv', help='data path')
    parser.add_argument('--test-data', type=str, default='dataset/nuscenes/test/data_all_camera_test.csv', help='data path')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.001, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/all_camera/try01', help='path to save checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    os.mkdir(cfg.save_path)
    # load_model = torch.load('weights/79.pt')
    load_model = MyNetwork()
    # total_params = sum(p.numel() for p in load_model.parameters())
    # print(f"Total parameters: {total_params}")
    # torch.cuda.empty_cache()

    train_val(cfg, load_model, device)
    # test(cfg, load_model, device)


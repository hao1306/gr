import sys
import os
import argparse
import torch
from network.resnet_onecamera import MyNetwork
from torch.utils.data import DataLoader, random_split
from dataset.dataloader_old import Dataset_onecamera, Dataset_onecamera_test
import utils
import cv2
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import math

def minADE_k(pre, truth):
    dis = 0
    n = pre.shape[0]
    m = pre.shape[1]
    count = 0
    for i in range(n):
        for j in range(m):
            dis += math.sqrt((pre[i, j, 0] - truth[i, j, 0]) ** 2 + (pre[i, j, 1] - truth[i, j, 1]) ** 2)
            count += 1
    output = dis / count

    return output

def minFDE_k(pre, truth):
    dis = 0
    n = pre.shape[0]
    m = pre.shape[1]
    count = 0
    for i in range(n):
        dis += math.sqrt((pre[i, -1, 0] - truth[i, -1, 0]) ** 2 + (pre[i, -1, 1] - truth[i, -1, 1]) ** 2)
        count += 1
    output = dis / count

    return output





def test(cfg, model_, device):

    dataset = Dataset_onecamera_test(data_path=cfg.test_data)
    # num_data = len(dataset)
    # aa = num_data - 5
    # test_dataset, _ = random_split(dataset, [5, aa])
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    #                               pin_memory=True)

    test_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    model = model_.to(device=device)

    fun1 = torch.nn.MSELoss()
    minADE_k = 
    loss_fun1 = 
    loss_fun1 = 

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
    parser.add_argument('--train-data', type=str, default='dataset/nuscenes/trian/data_one_camera.csv', help='data path')
    # parser.add_argument('--train-data', type=str, default='dataset/nuscenes/test/data_one_camera_test.csv', help='data path')
    parser.add_argument('--test-data', type=str, default='dataset/nuscenes/test/data_one_camera_test.csv', help='data path')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.001, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/one_camera/try01', help='path to save checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    device = torch.device('cuda:0')
    os.mkdir(cfg.save_path)
    # load_model = torch.load('weights/79.pt')
    load_model = MyNetwork()
    # total_params = sum(p.numel() for p in load_model.parameters())
    # print(f"Total parameters: {total_params}")
    # torch.cuda.empty_cache()

    train_val(cfg, load_model, device)
    # test(cfg, load_model, device)


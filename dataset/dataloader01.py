from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import glob
import cv2
import random
import argparse
import torch
import csv
import os
from pyquaternion import Quaternion
import sys
sys.path.append('/mnt/ssd/hao/')
from utils import preprocess, preprocess_all

class ResNetDataset(Dataset):
    def __init__(self, cfg, phase):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 使用当前帧和之前的input_n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        self.phase = phase
        self.cfg = cfg

        data_path = cfg.data_path + phase + '/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]]  # 
        self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]

        # 遍历文件夹中的所有 CSV 文件
        num = 0
        if self.image_n == 1:
            self.imgpath = []
            self.coo_in = []
            self.coo_out = []
            self.coo_orignal = []
            for filename in os.listdir(data_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_path, filename)

                    # 读取 CSV 文件
                    data = []
                    with open(file_path, 'r') as f:
                        aa = csv.reader(f)
                        for row in aa:
                            if '/CAM_FRONT/' in row[0]:
                                # num += 1
                                data.append(row)
                    len_data = len(data)

                    for i in range(len_data):
                        if (i + 1) > (self.input_n - 1) and (i + 1 + self.predict_n) <= len_data:
                            imgpath = [data[i - 3][0], data[i - 2][0], data[i - 1][0], data[i][0]]
                            a = float(data[i][1])
                            b = float(data[i][2])
                            #  yaw for input
                            q1 = Quaternion([float(data[i - 3][4]), float(data[i - 3][5]), float(data[i - 3][6]), float(data[i - 3][7])]).yaw_pitch_roll[0]
                            q2 = Quaternion([float(data[i - 2][4]), float(data[i - 2][5]), float(data[i - 2][6]), float(data[i - 2][7])]).yaw_pitch_roll[0]
                            q3 = Quaternion([float(data[i - 1][4]), float(data[i - 1][5]), float(data[i - 1][6]), float(data[i - 1][7])]).yaw_pitch_roll[0]
                            q4 = Quaternion([float(data[i][4]), float(data[i][5]), float(data[i][6]), float(data[i][7])]).yaw_pitch_roll[0]
                            # yaw for output
                            q5 = Quaternion([float(data[i + 1][4]), float(data[i + 1][5]), float(data[i + 1][6]), float(data[i + 1][7])]).yaw_pitch_roll[0]
                            q6 = Quaternion([float(data[i + 2][4]), float(data[i + 2][5]), float(data[i + 2][6]), float(data[i + 2][7])]).yaw_pitch_roll[0]
                            q7 = Quaternion([float(data[i + 3][4]), float(data[i + 3][5]), float(data[i + 3][6]), float(data[i + 3][7])]).yaw_pitch_roll[0]
                            q8 = Quaternion([float(data[i + 4][4]), float(data[i + 4][5]), float(data[i + 4][6]), float(data[i + 4][7])]).yaw_pitch_roll[0]
                            q9 = Quaternion([float(data[i + 5][4]), float(data[i + 5][5]), float(data[i + 5][6]), float(data[i + 5][7])]).yaw_pitch_roll[0]
                            q10 = Quaternion([float(data[i + 6][4]), float(data[i + 6][5]), float(data[i + 6][6]), float(data[i + 6][7])]).yaw_pitch_roll[0]
                            coo_in = [float(data[i - 3][1]) - a, float(data[i - 3][2]) - b, q1, \
                                      float(data[i - 2][1]) - a, float(data[i - 2][2]) - b, q2, \
                                      float(data[i - 1][1]) - a, float(data[i - 1][2]) - b, q3, \
                                      0, 0, q4]
                            coo_origial = [a, b, q4]
                            coo_out = [float(data[i + 1][1]) - a, float(data[i + 1][2]) - b, q5, \
                                       float(data[i + 2][1]) - a, float(data[i + 2][2]) - b, q6, \
                                       float(data[i + 3][1]) - a, float(data[i + 3][2]) - b, q7, \
                                       float(data[i + 4][1]) - a, float(data[i + 4][2]) - b, q8, \
                                       float(data[i + 5][1]) - a, float(data[i + 5][2]) - b, q9, \
                                       float(data[i + 6][1]) - a, float(data[i + 6][2]) - b, q10]
                            
                            self.imgpath.append(imgpath)
                            self.coo_out.append(coo_out)
                            self.coo_in.append(coo_in)
                            self.coo_orignal.append(coo_origial)
                            # print(len_data, len(self.coo_orignal))
        if self.image_n == 6:
            self.imgpath = []
            self.coo_in = []
            self.coo_out = []
            self.coo_orignal = []
            for filename in os.listdir(data_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_path, filename)

                    # 读取 CSV 文件
                    data = []
                    with open(file_path, 'r') as f:
                        aa = csv.reader(f)
                        for row in aa:
                            data.append(row)
                    len_data = len(data)

                    for i in range(len_data / 6):
                        if (i + 1) > (self.input_n - 1) and (i + 1 + self.predict_n) <= len_data:
                            imgpath = [data[i - 3][0], data[i - 2][0], data[i - 1][0], data[i][0]]
                            a = float(data[i][1])
                            b = float(data[i][2])
                            q1 = Quaternion([float(data[i - 3][4]), float(data[i - 3][5]), float(data[i - 3][6]), float(data[i - 3][7])]).yaw_pitch_roll[0]
                            q2 = Quaternion([float(data[i - 2][4]), float(data[i - 2][5]), float(data[i - 2][6]), float(data[i - 2][7])]).yaw_pitch_roll[0]
                            q3 = Quaternion([float(data[i - 1][4]), float(data[i - 1][5]), float(data[i - 1][6]), float(data[i - 1][7])]).yaw_pitch_roll[0]
                            q4 = Quaternion([float(data[i][4]), float(data[i][5]), float(data[i][6]), float(data[i][7])]).yaw_pitch_roll[0]
                            coo_in = [float(data[i - 3][1]) - a, float(data[i - 3][2]) - b, q1, \
                                      float(data[i - 2][1]) - a, float(data[i - 2][2]) - b, q2, \
                                      float(data[i - 1][1]) - a, float(data[i - 1][2]) - b, q3, \
                                      0, 0, q4]
                            coo_origial = [a, b]
                            coo_out = [float(data[i + 1][1]) - a, float(data[i + 1][2]) - b, \
                                       float(data[i + 2][1]) - a, float(data[i + 2][2]) - b, \
                                       float(data[i + 3][1]) - a, float(data[i + 3][2]) - b, \
                                       float(data[i + 4][1]) - a, float(data[i + 4][2]) - b, \
                                       float(data[i + 5][1]) - a, float(data[i + 5][2]) - b, \
                                       float(data[i + 6][1]) - a, float(data[i + 6][2]) - b]
                            
                            self.imgpath_one.append(imgpath)
                            self.coo_out.append(coo_out)
                            self.coo_in.append(coo_in)
                            self.coo_orignal.append(coo_origial)
        # print(len(self.coo_orignal))
        # print(num)


        
        
    def __len__(self):
        # print(len(self.coo_orignal))
        return len(self.coo_orignal)
    
    def img_process(self, imgpath):
        image = cv2.imread(self.data_root + imgpath)
        image = cv2.resize(image, self.image_size)
        if self.phase == 'train' or self.phase == 'mini_train':
                i = random.randint(0, self.crop_ij_max[0])
                j = random.randint(0, self.crop_ij_max[1])
                image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        else:
            image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]


        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = image.astype(np.float32)/255.0
        image = torch.from_numpy(image)
        return image
    
    def img_process6(self, imgpath):
        img = cv2.imread(self.data_root + imgpath)
        img = preprocess_all(img)
        img = img.astype(np.float32)/255.0
        return img
    
    def resize_combined_img(self, image):
        img = cv2.resize(image, (450, 800))
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):
        cfg = self.cfg

        if cfg.image_n == 1:
            imgpath = self.imgpath[idx]
            coo_in = self.coo_in[idx]
            coo_in = np.array(coo_in).reshape(4, 3)
            coo_out = self.coo_out[idx]
            coo_out = np.array(coo_out).reshape(6, 3)
            coo_orignal = self.coo_orignal[idx]
            image = [self.img_process(path_) for path_ in imgpath]
            batch = torch.stack(image)
            xy_in = torch.from_numpy(coo_in.astype(np.float32))
            xy_label = torch.from_numpy(coo_out.astype(np.float32))
            coo_orignal = torch.from_numpy(np.array(coo_orignal).astype(np.float32))

        if cfg.image_n == 6:
            None

        # print('xy_in size', xy_in.shape, 'xy label size', xy_label.shape)
        return batch, xy_in, xy_label, coo_orignal
    
class ResNetDataset_NOimage(Dataset):
    def __init__(self, cfg, phase):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 使用当前帧和之前的input_n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        self.phase = phase
        self.cfg = cfg

        data_path = cfg.data_path + phase + '/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        # 遍历文件夹中的所有 CSV 文件
        num = 0
        if self.image_n == 1:
            self.coo_in = []
            self.coo_out = []
            self.coo_orignal = []
            for filename in os.listdir(data_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(data_path, filename)

                    # 读取 CSV 文件
                    data = []
                    with open(file_path, 'r') as f:
                        aa = csv.reader(f)
                        for row in aa:
                            if '/CAM_FRONT/' in row[0]:
                                # num += 1
                                data.append(row)
                    len_data = len(data)

                    for i in range(len_data):
                        if (i + 1) > (self.input_n - 1) and (i + 1 + self.predict_n) <= len_data:
                            a = float(data[i][1])
                            b = float(data[i][2])
                            #  yaw for input
                            q1 = Quaternion([float(data[i - 3][4]), float(data[i - 3][5]), float(data[i - 3][6]), float(data[i - 3][7])]).yaw_pitch_roll[0]
                            q2 = Quaternion([float(data[i - 2][4]), float(data[i - 2][5]), float(data[i - 2][6]), float(data[i - 2][7])]).yaw_pitch_roll[0]
                            q3 = Quaternion([float(data[i - 1][4]), float(data[i - 1][5]), float(data[i - 1][6]), float(data[i - 1][7])]).yaw_pitch_roll[0]
                            q4 = Quaternion([float(data[i][4]), float(data[i][5]), float(data[i][6]), float(data[i][7])]).yaw_pitch_roll[0]
                            # yaw for output
                            q5 = Quaternion([float(data[i + 1][4]), float(data[i + 1][5]), float(data[i + 1][6]), float(data[i + 1][7])]).yaw_pitch_roll[0]
                            q6 = Quaternion([float(data[i + 2][4]), float(data[i + 2][5]), float(data[i + 2][6]), float(data[i + 2][7])]).yaw_pitch_roll[0]
                            q7 = Quaternion([float(data[i + 3][4]), float(data[i + 3][5]), float(data[i + 3][6]), float(data[i + 3][7])]).yaw_pitch_roll[0]
                            q8 = Quaternion([float(data[i + 4][4]), float(data[i + 4][5]), float(data[i + 4][6]), float(data[i + 4][7])]).yaw_pitch_roll[0]
                            q9 = Quaternion([float(data[i + 5][4]), float(data[i + 5][5]), float(data[i + 5][6]), float(data[i + 5][7])]).yaw_pitch_roll[0]
                            q10 = Quaternion([float(data[i + 6][4]), float(data[i + 6][5]), float(data[i + 6][6]), float(data[i + 6][7])]).yaw_pitch_roll[0]
                            coo_in = [float(data[i - 3][1]) - a, float(data[i - 3][2]) - b, q1, \
                                      float(data[i - 2][1]) - a, float(data[i - 2][2]) - b, q2, \
                                      float(data[i - 1][1]) - a, float(data[i - 1][2]) - b, q3, \
                                      0, 0, q4]
                            coo_origial = [a, b, q4]
                            coo_out = [float(data[i + 1][1]) - a, float(data[i + 1][2]) - b, q5, \
                                       float(data[i + 2][1]) - a, float(data[i + 2][2]) - b, q6, \
                                       float(data[i + 3][1]) - a, float(data[i + 3][2]) - b, q7, \
                                       float(data[i + 4][1]) - a, float(data[i + 4][2]) - b, q8, \
                                       float(data[i + 5][1]) - a, float(data[i + 5][2]) - b, q9, \
                                       float(data[i + 6][1]) - a, float(data[i + 6][2]) - b, q10]
                            
                            self.coo_out.append(coo_out)
                            self.coo_in.append(coo_in)
                            self.coo_orignal.append(coo_origial)
                            # print(len_data, len(self.coo_orignal))
               
    def __len__(self):
        # print(len(self.coo_orignal))
        return len(self.coo_orignal)
    

    def __getitem__(self, idx):
        cfg = self.cfg

        if cfg.image_n == 1:
            coo_in = self.coo_in[idx]
            coo_in = np.array(coo_in).reshape(4, 3)
            coo_out = self.coo_out[idx]
            coo_out = np.array(coo_out).reshape(6, 3)
            coo_orignal = self.coo_orignal[idx]
            xy_in = torch.from_numpy(coo_in.astype(np.float32))
            xy_label = torch.from_numpy(coo_out.astype(np.float32))
            coo_orignal = torch.from_numpy(np.array(coo_orignal).astype(np.float32))

        if cfg.image_n == 6:
            None

        # print('xy_in size', xy_in.shape, 'xy label size', xy_label.shape)
        return xy_in, xy_label, coo_orignal
    
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--data-root', type=str, default='dataset/nuscenes/trian/', help='path to save checkpoint')
    parser.add_argument('--data-path', type=str, default='dataset/nuscenes/image_scene/', help='path to save checkpoint')
    parser.add_argument('--image-n', type=int, default=1, help='num of cameras to be used')
    parser.add_argument('--input-n', type=int, default=4, help='num of images as input')
    parser.add_argument('--predict-n', type=int, default=6, help='num of points to be predicted')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='path to save checkpoint')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='path to save checkpoint')

    return parser.parse_args()
    
if __name__=='__main__':
    cfg = parse_cfg()
    train_data = ResNetDataset(cfg, phase='train')
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    i = 0
    for (image, xy_in, target, token) in train_dataloader:
        i += 1
        print('img', image.size())
        print('x in', xy_in.size())
        print('original', type(token), token.size())

        if i > 2:
            break

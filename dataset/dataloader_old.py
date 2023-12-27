import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import sys
sys.path.append('/mnt/ssd/hao/')
from utils import preprocess, preprocess_all

class Dataset_onecamera(Dataset):
    def __init__(self, data_path, phase='train'):

        # extract the data from csv file and store them in a list self.data
        self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        self.data = []
        with open(data_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)
    
    def img_process(self, imgpath):
        img = cv2.imread(imgpath)
        img = preprocess(img)
        img = img.astype(np.float32)/255
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):
        coordinate_input = []
        coordinate_label = []

        data_group = self.data[idx]

        coordinate_input += data_group[1:3] + data_group[4:6] + data_group[7:9] + data_group[10:12]
        coordinate_label = data_group[12:]

        imgpath1 = os.path.join(self.folder_dir, data_group[0])
        imgpath2 = os.path.join(self.folder_dir, data_group[3])
        imgpath3 = os.path.join(self.folder_dir, data_group[6])
        imgpath4 = os.path.join(self.folder_dir, data_group[9])
        
        img1 = self.img_process(imgpath1)
        img2 = self.img_process(imgpath2)
        img3 = self.img_process(imgpath3)
        img4 = self.img_process(imgpath4)

        batch = torch.stack([img1, img2, img3, img4])

        xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
        xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))

        odd_subtract = xy_in[-2].item()  # 奇数位要减去的数
        even_subtract = xy_in[-1].item()  # 偶数位要减去的数

        # 把坐标信息转换成相对于现在位置的坐标
        # 获取奇数位和偶数位的索引
        odd_indices01 = torch.arange(0, xy_in.size(0), step=2)
        even_indices01 = torch.arange(1, xy_in.size(0), step=2)
        odd_indices02 = torch.arange(0, xy_label.size(0), step=2)
        even_indices02 = torch.arange(1, xy_label.size(0), step=2)

        # 分别对奇数位和偶数位的数减去不同的值
        xy_in[even_indices01] -= even_subtract
        xy_in[odd_indices01] -= odd_subtract
        xy_label[even_indices02] -= even_subtract
        xy_label[odd_indices02] -= odd_subtract

        xy_label = torch.reshape(xy_label, (12, 2))

        return batch, xy_in, xy_label
    
class Dataset_onecamera_test(Dataset):
    def __init__(self, data_path, phase='test'):

        # extract the data from csv file and store them in a list self.data
        self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        self.data = []
        with open(data_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)


    def __len__(self):
        return len(self.data)
    
    def img_process(self, imgpath):
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (450, 800))
        img = img.astype(np.float32)/255
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):

        coordinate_input = []
        coordinate_label = []

        data_group = self.data[idx]

        coordinate_input += data_group[1:3] + data_group[4:6] + data_group[7:9] + data_group[10:12]
        coordinate_label = data_group[12:]

        imgpath1 = os.path.join(self.folder_dir, data_group[0])
        imgpath2 = os.path.join(self.folder_dir, data_group[3])
        imgpath3 = os.path.join(self.folder_dir, data_group[6])
        imgpath4 = os.path.join(self.folder_dir, data_group[9])
        
        img1 = self.img_process(imgpath1)
        img2 = self.img_process(imgpath2)
        img3 = self.img_process(imgpath3)
        img4 = self.img_process(imgpath4)

        batch = torch.stack([img1, img2, img3, img4])

        xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
        xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))

        odd_subtract = xy_in[-2].item()  # 奇数位要减去的数
        even_subtract = xy_in[-1].item()  # 偶数位要减去的数

        # 把坐标信息转换成相对于现在位置的坐标
        # 获取奇数位和偶数位的索引
        odd_indices01 = torch.arange(0, xy_in.size(0), step=2)
        even_indices01 = torch.arange(1, xy_in.size(0), step=2)
        odd_indices02 = torch.arange(0, xy_label.size(0), step=2)
        even_indices02 = torch.arange(1, xy_label.size(0), step=2)

        # 分别对奇数位和偶数位的数减去不同的值
        xy_in[even_indices01] -= even_subtract
        xy_in[odd_indices01] -= odd_subtract
        xy_label[even_indices02] -= even_subtract
        xy_label[odd_indices02] -= odd_subtract

        xy_label = torch.reshape(xy_label, (12, 2))

        return batch, xy_in, xy_label
    
class Dataset_allcamera(Dataset):
    def __init__(self, data_path, phase='train'):

        # extract the data from csv file and store them in a list self.data
        self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        self.data = []
        with open(data_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)


    def __len__(self):
        return len(self.data)
    
    def img_process(self, imgpath):
        img = cv2.imread(imgpath)
        img = preprocess_all(img)
        img = img.astype(np.float32)/255.0
        return img
    
    def resize_combined_img(self, image):
        img = cv2.resize(image, (450, 800))
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):

        coordinate_input = []
        coordinate_label = []
        img_path = []

        data_group = self.data[idx]

        for i in range(32): # extract input images' path and input coordiantes
            if i % 8 == 6 or i % 8 == 7:
                coordinate_input.append(data_group[i])
            else:
                img_path.append(data_group[i])

        coordinate_label = data_group[32:] # extrat label coordinates

        imagepath = [os.path.join(self.folder_dir, path) for path in img_path]
        images = [self.img_process(path_) for path_ in imagepath]
        # print('image size of images', images[0].shape)

        img = []
        for m in range(4):
            combined_image = np.zeros((2700, 3200, 3))
            for n in range(6):
                row = n // 2
                col = n % 2
                combined_image[(row * 900) : ((row + 1) * 900), (col * 1600) : ((col + 1) * 1600), :] = images[m * 6 + n]
            img.append(combined_image)

        imgs = [self.resize_combined_img(img_) for img_ in img]

        batch = torch.stack(imgs)

        xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
        xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))

        odd_subtract = xy_in[-2].item()  # 奇数位要减去的数
        even_subtract = xy_in[-1].item()  # 偶数位要减去的数

        # 把坐标信息转换成相对于现在位置的坐标
        # 获取奇数位和偶数位的索引
        odd_indices01 = torch.arange(0, xy_in.size(0), step=2)
        even_indices01 = torch.arange(1, xy_in.size(0), step=2)
        odd_indices02 = torch.arange(0, xy_label.size(0), step=2)
        even_indices02 = torch.arange(1, xy_label.size(0), step=2)

        # 分别对奇数位和偶数位的数减去不同的值
        xy_in[even_indices01] -= even_subtract
        xy_in[odd_indices01] -= odd_subtract
        xy_label[even_indices02] -= even_subtract
        xy_label[odd_indices02] -= odd_subtract

        xy_label = torch.reshape(xy_label, (12, 2))

        return batch, xy_in, xy_label
    
class Dataset_allcamera_test(Dataset):
    def __init__(self, data_path, phase='test'):

        # extract the data from csv file and store them in a list self.data
        self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        self.data = []
        with open(data_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)


    def __len__(self):
        return len(self.data)
    
    def img_process(self, imgpath):
        img = cv2.imread(imgpath)
        img = img.astype(np.float32)/255.0
        return img
    
    def resize_combined_img(self, image):
        img = cv2.resize(image, (450, 800))
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def __getitem__(self, idx):

        coordinate_input = []
        coordinate_label = []
        img_path = []

        data_group = self.data[idx]

        for i in range(32): # extract input images' path and input coordiantes
            if i % 8 == 6 or i % 8 == 7:
                coordinate_input.append(data_group[i])
            else:
                img_path.append(data_group[i])

        coordinate_label = data_group[32:] # extrat label coordinates

        imagepath = [os.path.join(self.folder_dir, path) for path in img_path]
        images = [self.img_process(path_) for path_ in imagepath]
        # print('image size of images', images[0].shape)

        img = []
        for m in range(4):
            combined_image = np.zeros((2700, 3200, 3))
            for n in range(6):
                row = n // 2
                col = n % 2
                combined_image[(row * 900) : ((row + 1) * 900), (col * 1600) : ((col + 1) * 1600), :] = images[m * 6 + n]
            img.append(combined_image)

        imgs = [self.resize_combined_img(img_) for img_ in img]

        batch = torch.stack(imgs)

        xy_in = torch.from_numpy(np.array(coordinate_input).astype(np.float32))
        xy_label = torch.from_numpy(np.array(coordinate_label).astype(np.float32))

        odd_subtract = xy_in[-2].item()  # 奇数位要减去的数
        even_subtract = xy_in[-1].item()  # 偶数位要减去的数

        # 把坐标信息转换成相对于现在位置的坐标
        # 获取奇数位和偶数位的索引
        odd_indices01 = torch.arange(0, xy_in.size(0), step=2)
        even_indices01 = torch.arange(1, xy_in.size(0), step=2)
        odd_indices02 = torch.arange(0, xy_label.size(0), step=2)
        even_indices02 = torch.arange(1, xy_label.size(0), step=2)

        # 分别对奇数位和偶数位的数减去不同的值
        xy_in[even_indices01] -= even_subtract
        xy_in[odd_indices01] -= odd_subtract
        xy_label[even_indices02] -= even_subtract
        xy_label[odd_indices02] -= odd_subtract

        xy_label = torch.reshape(xy_label, (12, 2))

        return batch, xy_in, xy_label

# datapath = 'dataset/nuscenes/trian/data_all_camera.csv'
# datapath_ = 'dataset/nuscenes/trian/data_all_camera.csv'
# data = Dataset_allcamera(datapath)
# data_ = Dataset_allcamera_test(datapath_)
# data01 = data[0]
# batch = data01[0]
# xyin = data01[1]
# xyout = data01[2]
# print(batch.shape)
# print(xyin.shape)
# print(xyout.shape)
# print(type(batch[0,0,0,0].item()))
# print(xyout)
# data01_ = data_[0]
# batch_ = data01_[0]
# xyin_ = data01_[1]
# xyout_ = data01_[2]
# print(batch_.shape)
# print(xyin_.shape)
# print(xyout_.shape)
# print(xyin_)
# print(xyout_)